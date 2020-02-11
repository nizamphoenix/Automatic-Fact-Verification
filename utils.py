import pickle

def save_object(obj, filepath):
    '''
    function to save a pickle object
    '''
    with open(filepath, 'wb') as output:  # Overwrites any existing file.
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)


def load_object(filepath):
    '''
    function to load a pickle object
    '''
    obj=None
    with (open(filepath, "rb")) as openfile:
        while True:
            try:
                obj = pickle.load(openfile)
            except EOFError:
                break
    return obj
