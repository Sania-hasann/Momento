import re
from typing import List

import regex as re
from llama_index.legacy.node_parser.text.token import TokenTextSplitter


class UnicodeSafeTokenTextSplitter(TokenTextSplitter):
    def _split(self, text: str, chunk_size: int) -> List[str]:
        """Break text into splits that are smaller than chunk size, preserving surrogate pairs."""
        if len(self._tokenizer(text)) <= chunk_size:
            return [text]

        # This regex pattern matches Unicode characters, including surrogate pairs
        pattern = re.compile(r"\X", re.UNICODE)

        splits = []
        current_split = ""
        current_tokens = 0

        for char in pattern.findall(text):
            char_tokens = len(self._tokenizer(char))
            if current_tokens + char_tokens > chunk_size:
                if current_split:
                    splits.append(current_split)
                current_split = char
                current_tokens = char_tokens
            else:
                current_split += char
                current_tokens += char_tokens

        if current_split:
            splits.append(current_split)

        return splits

    # The rest of the methods remain the same as in TokenTextSplitter


class MarkdownTokenTextSplitter(TokenTextSplitter):
    def _split_text(self, text: str, chunk_size: int) -> List[str]:
        lines = text.split("\n")
        chunks = []
        current_chunk = []
        current_tokens = 0
        headers = []
        table_headers = []
        in_table = False
        table_buffer = []  # Buffer to collect complete table
        numbered_pattern = re.compile(
            r"^\s*\d+\)"
        )  # Pattern for numbered items like 1), 2)

        def get_headers():
            return "\n".join(headers) + "\n" if headers else ""

        def get_table_headers():
            return "\n".join(table_headers) + "\n" if table_headers else ""

        def normalize_table_line(line: str) -> str:
            """Normalize table line by removing excessive whitespace while preserving structure."""
            if "|" not in line:
                return line

            # Split by pipe and process each cell
            cells = line.split("|")
            normalized_cells = []

            for cell in cells:
                # Preserve one space at beginning/end if content exists
                # This maintains table formatting while removing excess whitespace
                cell_content = cell.strip()
                if cell_content:
                    normalized_cell = f" {cell_content} "
                else:
                    normalized_cell = " "
                normalized_cells.append(normalized_cell)

            return "|".join(normalized_cells)

        def get_token_count(line: str) -> int:
            """Get token count for a line, with special handling for table lines."""
            if "|" in line:
                # Normalize table line before counting tokens
                normalized_line = normalize_table_line(line)
                return len(self._tokenizer(normalized_line))
            return len(self._tokenizer(line))

        def process_complete_table():
            """Process a complete table from the buffer - keep tables together as complete units."""
            nonlocal current_chunk, current_tokens
            if not table_buffer:
                return

            # Calculate total tokens for the complete table
            table_content = "\n".join(table_buffer)
            table_tokens = len(self._tokenizer(table_content))

            # Always keep the complete table together
            if current_tokens + table_tokens <= chunk_size:
                # Table fits in current chunk - add it
                current_chunk.extend(table_buffer)
                current_tokens += table_tokens
            else:
                # Table doesn't fit - start new chunk with complete table
                # This may exceed chunk_size, but preserves table integrity
                if current_chunk:
                    chunks.append(self._format_chunk(current_chunk))

                # Start new chunk with headers and complete table
                current_chunk = [get_headers(), get_table_headers()] + table_buffer
                current_tokens = len(self._tokenizer("\n".join(current_chunk)))

            table_buffer.clear()

        for line in lines:
            # Use the optimized token counting for tables
            line_tokens = get_token_count(line)

            # Handle headers
            if line.startswith("#"):
                level = len(line.split()[0])
                headers = headers[: level - 1] + [line.strip()]
                if current_chunk:
                    chunks.append(self._format_chunk(current_chunk))
                current_chunk = [get_headers()]
                current_tokens = len(self._tokenizer(get_headers()))
                continue

            # Table handling
            if "|" in line:
                # Normalize the table line to reduce unnecessary whitespace
                normalized_line = normalize_table_line(line)

                if not in_table:  # Start of table
                    in_table = True
                    # Store table headers (first row and separator)
                    table_headers = []
                    table_headers.append(normalized_line)  # Store normalized version
                    # Add header row to table buffer as well so it appears in content
                    table_buffer.append(normalized_line)
                    continue

                # Skip separator line but store it in table_headers
                if re.match(r"^[\s|:-]+$", line.strip()):
                    table_headers.append(normalized_line)  # Store normalized version
                    continue

                # Collect table rows in buffer for complete table processing
                table_buffer.append(normalized_line)

            else:  # Not a table line
                if in_table:
                    # End of table - process the complete table
                    process_complete_table()
                    in_table = False
                    table_headers = []

                if current_tokens + line_tokens > chunk_size:
                    chunks.append(self._format_chunk(current_chunk))
                    current_chunk = [get_headers(), line]
                    current_tokens = len(self._tokenizer("\n".join(current_chunk)))
                else:
                    current_chunk.append(line)
                    current_tokens += line_tokens

        # Process any remaining table at the end
        if in_table:
            process_complete_table()

        if current_chunk:
            chunks.append(self._format_chunk(current_chunk))

        return self._apply_overlap(chunks)

    def _format_chunk(self, chunk: List[str]) -> str:
        return "\n".join(filter(bool, chunk)).strip()

    def _apply_overlap(self, chunks: List[str]) -> List[str]:
        overlapped_chunks = []
        for i, chunk in enumerate(chunks):
            if i > 0:
                prev_chunk = chunks[i - 1]
                overlap_text, chunk = self._get_overlap(prev_chunk, chunk)
                chunk = overlap_text + chunk
            overlapped_chunks.append(chunk)
        return overlapped_chunks

    def _get_overlap(self, prev_chunk: str, current_chunk: str) -> tuple[str, str]:
        prev_lines = prev_chunk.split("\n")
        overlap_size = self.chunk_overlap
        overlap_text = []
        main_section = None
        table_headers = []

        # Extract main section header if present
        if prev_lines and prev_lines[0].startswith("#"):
            main_section = prev_lines[0]
            current_chunk = current_chunk.replace(main_section, "")
            prev_lines = prev_lines[1:]

        # Extract table headers if present
        in_table_header = False
        for i, line in enumerate(prev_lines):
            if "|" in line:
                if not in_table_header:
                    in_table_header = True
                    table_headers.append(line)
                elif re.match(r"^[\s|:-]+$", line.strip()):
                    table_headers.append(line)
                    break

        # Build overlap text
        for line in reversed(prev_lines):
            if len(self._tokenizer("\n".join(overlap_text))) >= overlap_size:
                break
            # Don't include table rows or headers in overlap unless they're table headers
            if not line.startswith("#") and ("|" not in line or line in table_headers):
                overlap_text.insert(0, line)

        # Construct final overlap
        final_overlap = []
        if main_section:
            final_overlap.append(main_section)
        if table_headers:
            final_overlap.extend(table_headers)
        if overlap_text:
            final_overlap.extend(overlap_text)

        return "\n".join(final_overlap) + "\n" if final_overlap else "", current_chunk
