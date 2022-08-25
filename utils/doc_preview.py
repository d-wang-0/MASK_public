from utils.readers import read_i2b2_file
import utils.spec_tokenizers

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

tag2color = {'ID': bcolors.HEADER, 'PHI': bcolors.UNDERLINE, 'NAME': bcolors.OKCYAN,
             'CONTACT': bcolors.OKGREEN, 'DATE': bcolors.WARNING, 'AGE': bcolors.FAIL,
             'PROFESSION': bcolors.BOLD, 'LOCATION': bcolors.OKBLUE}

def color_text(text, color):
    """
    Color text in terminal.
    """
    return color + text + bcolors.ENDC

def show_legend():
    """
    Prints legend for tags and colors.
    """
    for tag in tag2color:
        print(color_text(tag, tag2color[tag]))

def preview_token_seq(tokens):
    """
    Prints token sequence with tags colored.
    """
    out_string = ''
    for token, label in tokens:
        if label == 'O':
            out_string += f'({token}) '
        else:
            tag = label
            if tag in tag2color:
                out_string += color_text(f'({token}) ', tag2color[tag])
    return out_string

def preview_doc(document):
    """
    Prints document with tags colored.
    """
    documents = [document]
    tokens_labels = utils.spec_tokenizers.tokenize_to_seq(documents)

    tokens = []
    for token_label in tokens_labels:
        tokens+=token_label
    out_string = preview_token_seq(tokens)
    # print out tags in their color
    show_legend()
    print(out_string)

if __name__ == "__main__":
    preview_token_seq([('test','O'),('test','NAME')])
