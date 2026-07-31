
def words2characters(words):
    """
    This function converts a list of words into a list of characters.

    @param:
    words - a list of words

    @return:
    characters - a list of characters

    Every element of "words" should be converted to a str, then split into
    characters, each of which is separately appended to "characters." For 
    example, if words==['hello', 1.234, True], then characters should be
    ['h', 'e', 'l', 'l', 'o', '1', '.', '2', '3', '4', 'T', 'r', 'u', 'e']
    """
    characters = []
    for word in words:
        word_str = str(word)
        for char in word_str:
            characters.append(char)
    return characters

def cancellation(input_list, stop_word):
    """
    Copy elements from input_list into output_list one by one.
    If an element equals stop_word, stop and return output_list.

    @params:
    input_list - a list of elements
    stop_word - the element that triggers stopping

    @return:
    output_list - list of elements copied before stop_word was found
    """
    output_list = []
    for element in input_list:
        if element == stop_word:
            break
        output_list.append(element)
    return output_list

def copy_all_but_skip_word(input_list, skip_word):
    """
    Copy all elements from input_list into output_list,
    except elements equal to skip_word (which are skipped).

    @params:
    input_list - a list of elements
    skip_word - the element to skip

    @return:
    output_list - list of elements except those equal to skip_word
    """
    output_list = []
    for element in input_list:
        if element == skip_word:
            continue
        output_list.append(element)
    return output_list

def my_average(input_list):
    """
    Calculate the average of all numeric elements in input_list.

    @param:
    input_list - a non-empty list of numbers

    @return:
    The average value as a float
    """
    total = 0
    count = 0
    for element in input_list:
        total = total + element
        count = count + 1
    return total / count

