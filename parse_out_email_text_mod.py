#!/usr/bin/python

from nltk.stem.snowball import SnowballStemmer
import string
import re

def parseOutText(f):
    """ given an opened email file f, parse out all text below the
        metadata block at the top
        (in Part 2, you will also add stemming capabilities)
        and return a string that contains all the words
        in the email (space-separated) 
        
        example use case:
        f = open("email_file_name.txt", "r")
        text = parseOutText(f)
        
        """


    f.seek(0)  ### go back to beginning of file (annoying)
    all_text = f.read()

    ### split off metadata
    content = all_text.split("X-FileName:")
    words = ""
    from_email = ""
    if len(content) > 1:
        #get sender email
        try:
            from_meta = re.search('\nFrom:.+\n', content[0])
            from_email = re.search('[A-Za-z0-9._-]+[@].+[.].+$', from_meta.group(0)).group(0)
        except:
            from_email = None
            
        
        ### remove punctuation
        text_string = content[1].translate(string.maketrans("", ""), string.punctuation)

        ### project part 2: comment out the line below
        words = text_string.split()

        ### split the text string into individual words, stem each word,
        ### and append the stemmed word to words (make sure there's a single
        ### space between each stemmed word)

        stemmer = SnowballStemmer("english")
        words = [stemmer.stem(word) for word in words]
        words = ' '.join(words)

    return from_email, words

    

def main():
    ff = open("D:/Cloud\Google Drive/Udacity - data Analyst/ud120-projects/text_learning/test_email.txt", "r")
    from_email, text = parseOutText(ff)
    print(from_email)
    print text



if __name__ == '__main__':
    main()

