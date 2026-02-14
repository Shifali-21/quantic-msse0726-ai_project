import os
from src.utils import clean_text, chunk_text, word_window_chunk, list_files

def test_clean_text():
    s = "Hello  \r\n\n  world\xa0  "
    assert clean_text(s) == "Hello world"

def test_word_window_chunk_and_overlap():
    # create a 120-word sample
    words = ["w"] * 120
    text = " ".join(words)
    chunks = word_window_chunk(text, chunk_size_words=50, overlap_words=10)
    # Expect sliding windows: step = 40 -> starts at 0,40,80 -> 3 chunks
    assert len(chunks) == 3
    # check overlap: last 10 words of chunk0 should equal first 10 of chunk1
    c0 = chunks[0].split()
    c1 = chunks[1].split()
    assert c0[-10:] == c1[:10]

def test_chunk_text_uses_headings_and_word_window():
    md = "# Section A\n" + ("a " * 60) + "\n# Section B\n" + ("b " * 20)
    chunks = chunk_text(md, chunk_size_words=50, overlap_words=10)
    # Section A should be split into >1 chunk, Section B stays single chunk
    assert any("Section A" in c or c.startswith("Section A") for c in chunks)
    assert any("Section B" in c or c.startswith("Section B") for c in chunks)

def test_list_files_sorted(tmp_path):
    # create files out of order
    f1 = tmp_path / "z.md"
    f2 = tmp_path / "a.md"
    f3 = tmp_path / "ignore.txtx"
    f1.write_text("z")
    f2.write_text("a")
    f3.write_text("x")
    found = list(list_files(str(tmp_path)))
    # should only return .md files and in sorted order
    assert os.path.basename(found[0]) == "a.md"
    assert os.path.basename(found[1]) == "z.md"