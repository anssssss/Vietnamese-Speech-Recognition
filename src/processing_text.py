# encodes text data

# Vietnamese characters
chars = 'AĂÂBCDĐEÊGHIKLMNOÔƠPQRSTUƯVXYÁÀẢÃẠĂẮẰẲẴẶÂẤẦẨẪẬĐÉÈẺẼẸÊẾỀỂỄỆÍÌỈĨỊÓÒỎÕỌÔỐỒỔỖỘƠỚỜỞỠỢÚÙỦŨỤƯỨỪỬỮỰÝỲỶỸỴ '

# encodes characters to numbers
def encode(text):
    encoded = []
    for char in text:
        encoded.append(chars.find(char))
    return encoded

# decodes numbers to characters
def decode(encoded):
    text = []
    for no in encoded:
        if no < 97:
            text.append(chars[no])
    return text

# writes encoded text to file
def write(set):
    
    # data path
    path = './data/'

    with open(path + 'encoded/' + set + '.txt', 'w') as out:
        with open(path + 'vivos/' + set + '/prompts.txt', encoding='utf-8') as inp:

            # reads raw text file as a list
            content = inp.readlines()
            for prompt in content:
                for char in encode(prompt[prompt.index(' ') + 1:]):
                    out.write(str(char) + ' ')
                out.write('\n')

if __name__ == '__main__':
    write('train')
    write('test')