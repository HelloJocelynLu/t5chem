import os
import tempfile
import pytest
from t5chem.data_utils import count_lines, python_chunked_count


class TestPythonChunkedCount:
    """Test the python_chunked_count function."""
    
    def test_empty_file(self):
        """Test counting lines in an empty file."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
            temp_path = f.name
        
        try:
            result = python_chunked_count(temp_path)
            assert result == 0, "Empty file should have 0 lines"
        finally:
            os.unlink(temp_path)
    
    def test_single_line_with_newline(self):
        """Test file with single line ending with newline."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
            f.write("line1\n")
            temp_path = f.name
        
        try:
            result = python_chunked_count(temp_path)
            assert result == 1, "File with one line and trailing newline should have 1 line"
        finally:
            os.unlink(temp_path)
    
    def test_single_line_without_newline(self):
        """Test file with single line not ending with newline."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
            f.write("line1")
            temp_path = f.name
        
        try:
            result = python_chunked_count(temp_path)
            assert result == 1, "File with one line without trailing newline should have 1 line"
        finally:
            os.unlink(temp_path)
    
    def test_multiple_lines_with_newline(self):
        """Test file with multiple lines ending with newline."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
            f.write("line1\nline2\nline3\n")
            temp_path = f.name
        
        try:
            result = python_chunked_count(temp_path)
            assert result == 3, "File with 3 lines and trailing newline should have 3 lines"
        finally:
            os.unlink(temp_path)
    
    def test_multiple_lines_without_newline(self):
        """Test file with multiple lines not ending with newline."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
            f.write("line1\nline2\nline3")
            temp_path = f.name
        
        try:
            result = python_chunked_count(temp_path)
            assert result == 3, "File with 3 lines without trailing newline should have 3 lines"
        finally:
            os.unlink(temp_path)
    
    def test_large_file_chunked_reading(self):
        """Test chunked reading with a file larger than buffer size."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
            # Create a file with 10000 lines, larger than typical buffer
            for i in range(10000):
                f.write(f"This is line {i} with some content\n")
            temp_path = f.name
        
        try:
            result = python_chunked_count(temp_path, bufsize=1024)  # Small buffer to test chunking
            assert result == 10000, "Large file should be counted correctly with chunked reading"
        finally:
            os.unlink(temp_path)
    
    def test_large_file_without_trailing_newline(self):
        """Test large file without trailing newline."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
            # Write 9999 lines with newlines, and last line without
            for i in range(9999):
                f.write(f"This is line {i}\n")
            f.write("This is the last line without newline")
            temp_path = f.name
        
        try:
            result = python_chunked_count(temp_path, bufsize=1024)
            assert result == 10000, "Large file without trailing newline should count correctly"
        finally:
            os.unlink(temp_path)
    
    def test_file_with_only_newlines(self):
        """Test file containing only newline characters."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
            f.write("\n\n\n")
            temp_path = f.name
        
        try:
            result = python_chunked_count(temp_path)
            assert result == 3, "File with 3 newlines should have 3 lines"
        finally:
            os.unlink(temp_path)
    
    def test_binary_content(self):
        """Test file with binary content containing newlines."""
        with tempfile.NamedTemporaryFile(mode='wb', delete=False) as f:
            f.write(b"binary\ndata\nwith\nnewlines\n")
            temp_path = f.name
        
        try:
            result = python_chunked_count(temp_path)
            assert result == 4, "Binary file with 4 newlines should have 4 lines"
        finally:
            os.unlink(temp_path)


class TestCountLines:
    """Test the count_lines function (which uses wc -l or falls back to python_chunked_count)."""
    
    def test_empty_file(self):
        """Test counting lines in an empty file."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
            temp_path = f.name
        
        try:
            result = count_lines(temp_path)
            assert result == 0, "Empty file should have 0 lines"
        finally:
            os.unlink(temp_path)
    
    def test_single_line_with_newline(self):
        """Test file with single line ending with newline."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
            f.write("line1\n")
            temp_path = f.name
        
        try:
            result = count_lines(temp_path)
            assert result == 1, "File with one line and trailing newline should have 1 line"
        finally:
            os.unlink(temp_path)
    
    def test_single_line_without_newline(self):
        """Test file with single line not ending with newline."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
            f.write("line1")
            temp_path = f.name
        
        try:
            result = count_lines(temp_path)
            assert result == 1, "File with one line without trailing newline should have 1 line"
        finally:
            os.unlink(temp_path)
    
    def test_multiple_lines_with_newline(self):
        """Test file with multiple lines ending with newline."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
            f.write("line1\nline2\nline3\n")
            temp_path = f.name
        
        try:
            result = count_lines(temp_path)
            assert result == 3, "File with 3 lines and trailing newline should have 3 lines"
        finally:
            os.unlink(temp_path)
    
    def test_multiple_lines_without_newline(self):
        """Test file with multiple lines not ending with newline."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
            f.write("line1\nline2\nline3")
            temp_path = f.name
        
        try:
            result = count_lines(temp_path)
            assert result == 3, "File with 3 lines without trailing newline should have 3 lines"
        finally:
            os.unlink(temp_path)
    
    def test_consistency_with_python_chunked_count(self):
        """Test that count_lines gives same result as python_chunked_count."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
            f.write("line1\nline2\nline3\nline4\nline5")
            temp_path = f.name
        
        try:
            count_lines_result = count_lines(temp_path)
            python_chunked_result = python_chunked_count(temp_path)
            assert count_lines_result == python_chunked_result, \
                "count_lines and python_chunked_count should give same result"
        finally:
            os.unlink(temp_path)
    
    def test_large_file(self):
        """Test counting lines in a large file."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
            expected_lines = 5000
            for i in range(expected_lines):
                f.write(f"Line {i} with some content here\n")
            temp_path = f.name
        
        try:
            result = count_lines(temp_path)
            assert result == expected_lines, f"Large file should have {expected_lines} lines"
        finally:
            os.unlink(temp_path)
    
    def test_special_characters(self):
        """Test file with special characters and unicode."""
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', delete=False) as f:
            f.write("Line with émojis 🔬\n")
            f.write("Line with symbols: @#$%^&*()\n")
            f.write("Last line")
            temp_path = f.name
        
        try:
            result = count_lines(temp_path)
            assert result == 3, "File with special characters should have 3 lines"
        finally:
            os.unlink(temp_path)
    
    def test_nonexistent_file(self):
        """Test that nonexistent file raises appropriate error."""
        with pytest.raises(Exception):
            count_lines("/nonexistent/path/to/file.txt")
    
    def test_real_world_scenario(self):
        """Test scenario similar to actual usage in data_utils.py."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.source') as f:
            # Simulate a realistic dataset file
            reactions = [
                "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O>>CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",
                "C1=CC=C(C=C1)C(=O)O>>C1=CC=C(C=C1)C(=O)O",
                "CC(=O)OC1=CC=CC=C1C(=O)O>>CC(=O)OC1=CC=CC=C1C(=O)O",
            ]
            for reaction in reactions:
                f.write(reaction + "\n")
            temp_path = f.name
        
        try:
            result = count_lines(temp_path)
            assert result == 3, "Realistic dataset file should have 3 lines"
        finally:
            os.unlink(temp_path)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
