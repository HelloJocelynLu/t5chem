from typing import List, Optional

import torch
from torch import nn
from transformers import T5Config, T5ForConditionalGeneration
from transformers.modeling_outputs import Seq2SeqLMOutput


class T5ForProperty(T5ForConditionalGeneration):
    r"""
    T5Chem Model with a `language modeling` head on top. 
    Args:
        head_type (:obj:`string`, `optional`, defaults to `None`):
            The output layer type, can be either `Classification` or `Regression`,
            If `None`, no output layer would be used. (Will directly output decoder vectors)
        n_layer (:obj:`int`, `optional`, defaults to 0):
            The number of layers prepended to output layer.
        num_classes (:obj:`int`, `optional`, defaults to 500):
            The number of classes used for classification task, would be ignore for 
            regression task.
    """
    _keys_to_ignore_on_load_missing: List[str] = [
        r"encoder\.embed_tokens\.weight",
        r"decoder\.embed_tokens\.weight"
        r"lm_head\.0\.weight",
        r"lm_head\.0\.bias",
    ]
    _keys_to_ignore_on_load_unexpected: List[str] = [
        r"decoder\.block\.0\.layer\.1\.EncDecAttention\.relative_attention_bias\.weight",
        r"lm_head\.weight",
    ]
    def __init__(
        self, 
        config: T5Config, 
        head_type: Optional[str]=None, 
        n_layer: Optional[int]=None, 
        num_classes: Optional[int]=None,
        ) -> None:
        super().__init__(config)
        self.head_type = head_type if head_type else getattr(config, "head_type", None)
        n_layer = n_layer if n_layer else getattr(config, "n_layer", 0)
        lm_head_layers: List[nn.Module] = []
        unit_layer: List[nn.Module] = [
                nn.Linear(config.d_model, config.d_model),
                nn.ReLU(),
                ]
        for i in range(n_layer):
            lm_head_layers.extend(unit_layer)
        if not self.head_type:
            pass
        elif self.head_type == "classification":
            num_classes = num_classes if num_classes else getattr(config, "num_classes", 500)
            lm_head_layers.extend([
                nn.Linear(config.d_model, num_classes)
                ])
            self.config.num_classes = num_classes # type: ignore
        else:
            assert self.head_type == "regression", \
                "Only `classification` or `regression` are currently supported for output layer"
            lm_head_layers.extend([
                nn.Linear(config.d_model, 2),
                nn.LogSoftmax(dim=-1)
                ])
        self.set_output_embeddings(nn.Sequential(*lm_head_layers))
        self.config.tie_word_embeddings = False
        self.config.head_type = self.head_type # type: ignore
        self.config.n_layer = n_layer # type: ignore

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        encoder_outputs=None,
        past_key_values=None,
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):

        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if head_mask is not None and decoder_head_mask is None:
            if self.config.num_layers == self.config.num_decoder_layers:
                decoder_head_mask = head_mask

        # Encode if needed (training, first prediction pass)
        if encoder_outputs is None:
            # Convert encoder inputs in embeddings if needed
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

        hidden_states = encoder_outputs[0]

        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)

        if decoder_input_ids is None and decoder_inputs_embeds is None:
            # Initialize decoder by providing start_token_id
            decoder_input_ids = torch.full((input_ids.size(0),1),
                                            self.config.decoder_start_token_id,
                                            dtype=torch.long,
                                            device=input_ids.device)

        # If decoding with past key value states, only the last tokens
        # should be given as an input
        if past_key_values is not None:
            assert labels is None, "Decoder should not use cached key value states when training."
            if decoder_input_ids is not None:
                decoder_input_ids = decoder_input_ids[:, -1:]
            if decoder_inputs_embeds is not None:
                decoder_inputs_embeds = decoder_inputs_embeds[:, -1:]

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)
            hidden_states = hidden_states.to(self.decoder.first_device)
            if decoder_input_ids is not None:
                decoder_input_ids = decoder_input_ids.to(self.decoder.first_device)
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.decoder.first_device)
            if decoder_attention_mask is not None:
                decoder_attention_mask = decoder_attention_mask.to(self.decoder.first_device)
        # Decode
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            inputs_embeds=decoder_inputs_embeds,
            past_key_values=past_key_values,
            encoder_hidden_states=hidden_states,
            encoder_attention_mask=attention_mask,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = decoder_outputs[0]

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.encoder.first_device)
            self.lm_head = self.lm_head.to(self.encoder.first_device)
            sequence_output = sequence_output.to(self.lm_head.weight.device)

        if self.config.tie_word_embeddings:
            # Rescale output before projecting on vocab
            # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/transformer.py#L586
            sequence_output = sequence_output * (self.model_dim ** -0.5)
        if self.lm_head:
            lm_logits = self.lm_head(sequence_output.view(sequence_output.size()[0], -1))
        else:
            lm_logits = sequence_output.view(sequence_output.size()[0], -1)
            labels = None

        loss = None
        if labels is not None:
            if self.head_type == "classification":
                loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
                labels = labels.long()
                loss = loss_fct(lm_logits, labels.view(-1))
                lm_logits = torch.argmax(lm_logits, axis=-1)
            else:
                loss_fct = nn.KLDivLoss(reduction='batchmean')
                smoothed_label = torch.stack([(100-labels), labels], dim=1)/100
                loss = loss_fct(lm_logits, smoothed_label.view(-1,2))
                lm_logits = torch.exp(lm_logits[:,-1])*100

        if not return_dict:
            output = (lm_logits,) + decoder_outputs[1:] + encoder_outputs
            return ((loss,) + output) if loss is not None else output

        return Seq2SeqLMOutput(
            loss=loss,
            logits=lm_logits,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )

    def freeze_body(self):
        for name, param in self.named_parameters():
            if not (name.startswith('lm_head')):
                param.requires_grad = False
