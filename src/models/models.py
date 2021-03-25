import torch
from torch import nn

from copy import deepcopy
from collections import OrderedDict
from sys import stderr

# for type hint
from torch import Tensor
from transformers import T5ForConditionalGeneration
from transformers.modeling_outputs import Seq2SeqLMOutput

class EMA(nn.Module):
    def __init__(self, model: nn.Module, decay: float):
        super().__init__()
        self.decay = decay

        self.model = model
        self.config = self.model.config
        self.shadow = deepcopy(self.model)

        for param in self.shadow.parameters():
            param.detach_()

    @torch.no_grad()
    def update(self):
        if not self.training:
            print("EMA update should only be called during training", file=stderr, flush=True)
            return

        model_params = OrderedDict(self.model.named_parameters())
        shadow_params = OrderedDict(self.shadow.named_parameters())

        # check if both model contains the same set of keys
        assert model_params.keys() == shadow_params.keys()

        for name, param in model_params.items():
            # see https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage
            # shadow_variable -= (1 - decay) * (shadow_variable - variable)
            shadow_params[name].sub_((1. - self.decay) * (shadow_params[name] - param))

        model_buffers = OrderedDict(self.model.named_buffers())
        shadow_buffers = OrderedDict(self.shadow.named_buffers())

        # check if both model contains the same set of keys
        assert model_buffers.keys() == shadow_buffers.keys()

        for name, buffer in model_buffers.items():
            # buffers are copied
            shadow_buffers[name].copy_(buffer)

    def forward(self, **inputs):
        if self.training:
            return self.model(**inputs)
        else:
            return self.shadow(**inputs)

class T5ForSoftLabel(T5ForConditionalGeneration):
    _keys_to_ignore_on_load_missing = [
        r"encoder\.embed_tokens\.weight",
        r"decoder\.embed_tokens\.weight"
        r"lm_head\.0\.weight",
        r"lm_head\.0\.bias",
    ]
    _keys_to_ignore_on_load_unexpected = [
        r"decoder\.block\.0\.layer\.1\.EncDecAttention\.relative_attention_bias\.weight",
        r"lm_head\.weight",
    ]
    def __init__(self, config, loss_type=None, n_layer=None, num_classes=None):
        super().__init__(config)
        loss_type = loss_type if loss_type else getattr(config, "loss_type", None)
        n_layer = n_layer if n_layer else getattr(config, "n_layer", 1)
        lm_head_layers = []
        unit_layer = [
                nn.Linear(config.d_model, config.d_model),
                nn.BatchNorm1d(config.d_model),
                nn.ReLU(),
                nn.Dropout(config.dropout_rate)
                ]
        for i in range(n_layer-1):
            lm_head_layers.extend(unit_layer)
        if loss_type == "KLD":
            lm_head_layers.extend([
                nn.Linear(config.d_model, 2),
                nn.LogSoftmax(dim=-1)
                ])
        elif loss_type == "MSE":
            lm_head_layers.extend([
                nn.Linear(config.d_model, 1),
                nn.Sigmoid()
                ])
        else:
            num_classes = num_classes if num_classes else getattr(config, "num_classes", None)
            lm_head_layers.extend([
                nn.Linear(config.d_model, num_classes)
                ])
            self.config.num_classes = num_classes
        self.set_output_embeddings(nn.Sequential(*lm_head_layers))
        self.config.tie_word_embeddings = False
        self.config.loss_type = loss_type
        self.config.n_layer = n_layer

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
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

        # FutureWarning: head_mask was separated into two input args - head_mask, decoder_head_mask
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

        if labels is not None and decoder_input_ids is None and decoder_inputs_embeds is None:
            # get decoder inputs from shifting lm labels to the right
            # decoder_input_ids = self._shift_right(labels)
            decoder_input_ids = torch.full((labels.size(0),1),
                                            self.config.decoder_start_token_id,
                                            dtype=torch.long,
                                            device=labels.device)

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
            encoder_head_mask=head_mask,
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

        lm_logits = self.lm_head(sequence_output.view(sequence_output.size()[0], -1))

        loss = None
        if labels is not None:
            if self.config.loss_type == "KLD":
                loss_fct = nn.KLDivLoss(reduction='batchmean')
                smoothed_label = torch.stack([(100-labels), labels], dim=1)/100
                loss = loss_fct(lm_logits, smoothed_label.view(-1,2))
                lm_logits = torch.exp(lm_logits[:,-1])*100
            elif self.config.loss_type == "MSE":
                loss_fct = nn.MSELoss()
                smoothed_label = labels/100
                loss = loss_fct(lm_logits.view(-1), smoothed_label.view(-1))
                lm_logits = lm_logits.view(-1)*100
            else:
                loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
                labels = labels.long()
                # scale_factor = 100//self.config.num_classes
                # loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1)/scale_factor)
                loss = loss_fct(lm_logits, labels.view(-1))
                # lm_logits = torch.argmax(lm_logits, axis=-1).float().squeeze()*scale_factor+scale_factor/2
                lm_logits = torch.argmax(lm_logits, axis=-1)
            # TODO(thom): Add z_loss https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/layers.py#L666

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
            if not name.startswith('lm_head'):
                param.requires_grad = False
