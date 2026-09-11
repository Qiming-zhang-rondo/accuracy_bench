from accuracy_checker.token_display import token_display_text


def test_token_display_marks_whitespace_and_controls_without_remapping_unicode():
    assert token_display_text(" a\n\t\u200b\x00") == "·a↵⇥<U+200B><U+0000>"
    assert token_display_text("Ā") == "Ā"
    assert token_display_text("", 123) == "#123"
