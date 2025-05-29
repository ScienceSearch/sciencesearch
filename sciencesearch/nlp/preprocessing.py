from string import punctuation
import regex as re


def clean_text(text: str = None, filepath: str = None):
    """ Text preprocessing to remove extraneous symbols
        Args:
            text: a string of text
            filepath: a filepath to a text file

        Return:
            Cleaned text
        """
    txt = text
    if filepath is not None:
        with open(filepath, 'r') as file:
            txt = file.read()

    # 1. Remove newlines, tabs, and carriage returns
    # =============================
    no_breaks_txt = re.sub(r"[\n\t\r]", " ", txt)
    # no_breaks_txt = re.sub("\n", " ", txt)
    # no_breaks_txt = re.sub("\t", " ", no_breaks_txt)
    # no_tab_txt = re.sub("\r", " ", no_breaks_txt)

    # 2. Remove URLs
    # =============================
    url_string = r"\S*https?://\S*"
    url_replacement = " "
    no_url_txt = re.sub(url_string, url_replacement, no_breaks_txt)

    url_string = r"\S*www\.\S*"
    url_replacement = " "
    no_url_txt = re.sub(url_string, url_replacement, no_url_txt)
    
    # 3. Remove Punctuation
    #  Define punctuation to remove (excluding ., -, :, ;)
    # punct = '!"#$%&\'()*+/<=>?@[\\]^_`{|}~:;,•’”“\/'
    # =============================
    remove = punctuation + "•" + "–" + "’" + "”" + "“"
    remove = remove.replace(".", "")  # don't remove full-stops
    remove = remove.replace(",", "")  # don't remove commas
    remove = remove.replace("’", "")  # don't remove apostrophes
    remove = remove.replace("'", "")  # don't remove apostrophes


    remove = remove.replace(
        "-", ""
    )  # don't blanket-remove hyphens, will be removed after
    # remove = remove.replace(":", "")  # don't remove colons
    # remove = remove.replace(";", "")  # don't remove semi-colons
    punct_pattern = r"[{}]".format(remove)  # create the pattern
    punct_pattern = r"[{}]".format(remove)  # create the pattern
    no_punct_txt = re.sub(punct_pattern, " ", no_url_txt)

    # Only remove dashes with spaces before or after them. This way abbreviation replacements with dashes are preserved.
    # Remove ellipsis (...)
    no_punct_txt = re.sub(
            r"(\s?)\.{2,}",
            r".",
            re.sub(r"(\s?)[-](\s)", r" ", re.sub(r"(\s)[-](\s?)", r" ", no_punct_txt)),
        )
    
    # 4. Remove numbers
    # =============================
    # Remove all decimal numbers
    number_pattern = r"\s\d+(?:\.\d+)?"  # \s\d+'
    number_replacement = " "  # ' nos'
    no_digit_txt = re.sub(number_pattern, number_replacement, no_punct_txt)

    # 5.  Remove all numbers preceeded by a non-alphanumeric character, e.g. a dash. This will leave chemical formulas in.
    number_pattern = r"[^a-zA-Z0-9]+\d+"

    number_replacement = " "  #
    no_digit_txt = re.sub(number_pattern, number_replacement, no_digit_txt)

    # 6. Remove non-ascii characters
    # ================================
    no_ascii_text = no_digit_txt.encode("ascii", "ignore").decode()

    # 7. Remove lone punctuation
    no_lone_punct = re.sub(r'(?<=\s)[^\w\s]+(?=\s)', '', no_ascii_text)
    
    print(f"clean text: {no_ascii_text}")
    print(f"no lone punct {no_lone_punct}")

    # 8. Remove extra whitespaces
    # =============================
    whitespace_pattern = r"\s+"
    no_whitespace_txt = re.sub(whitespace_pattern, " ", no_lone_punct)
    no_whitespace_txt = no_whitespace_txt.rstrip()
    no_whitespace_txt = no_whitespace_txt.lstrip()
    print(f"no whitespace text {no_whitespace_txt}")

    return no_whitespace_txt
