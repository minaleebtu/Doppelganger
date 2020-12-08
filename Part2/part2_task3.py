import sys
import subprocess


def do_work():
    args = sys.argv
    option = args[1:2]  # First element of args is the file name
    keyword = args[2:]

    if len(option) == 0:
        print("You have not passed any commands in!")
        print("Usage: part2_task3.py [OPTIONS]\n")
        print("Options:\n\t-h\t\t print overview information about this toolbox")
        print("\t-s [KEYWORD]\t\tstart toolbox in different operation modes")
        print("\n(valid keyword:\n\tarticle -> collect article information from zeit.de")
        print("\tcomment -> collect comment according to articles from zeit.de")
        print("\tusercomm -> collect comments from 50 users who wrote at least 100 different comments")
        print("\tpreprocess -> pre-processing of the collected user comments")
        print("\tfeature -> extract a set of features from the collected user comments)")
    else:
        for opt in option:
            if opt == '-h' or opt == '-help':
                print("This toolbox is for scrapying article and comment information from the zeit.de which is a news webpage. \nMoreover, data from the webpage is able for preprocessing and generating features.")
            elif opt == '-s':
                if keyword:
                    for key in keyword:
                        if key == 'article':
                            subprocess.call(['python', '../Part1/part1_task1.py'])
                        elif key == 'comment':
                            subprocess.call(['python', '../Part1/part1_task2.py'])
                        elif key == 'usercomm':
                            subprocess.call(['python', '../Part1/part1_task3.py'])
                        elif key == 'preprocess':
                            subprocess.call(['python', 'part2_task1.py'])
                        elif key == 'feature':
                            subprocess.call(['python', 'part2_task2.py'])
                        else:
                            print("Please enter valid keyword")
                            print("valid keyword:\n\t\tarticle -> collect article information from zeit.de")
                            print("\t\tcomment -> collect comment according to articles from zeit.de")
                            print("\t\tusercomm -> collect comments from 50 users who wrote at least 100 different comments")
                            print("\t\tpreprocess -> pre-processing of the collected user comments")
                            print("\t\tfeature -> extract a set of features from the collected user comments")
                else:
                    print("Please enter valid keyword")
                    print("valid keyword:\n\t\tarticle -> collect article information from zeit.de")
                    print("\t\tcomment -> collect comment according to articles from zeit.de")
                    print("\t\tusercomm -> collect comments from 50 users who wrote at least 100 different comments")
                    print("\t\tpreprocess -> pre-processing of the collected user comments")
                    print("\t\tfeature -> extract a set of features from the collected user comments")
            else:
                print("Unrecognised argument.")
                print("Usage: part2_task3.py [OPTIONS]\n\n")
                print("Options:\n\t\t-h\t\t\t print overview information about this toolbox")
                print("\t\t-s [KEYWORD]\t\t\tstart toolbox in different operation modes")
                print("(valid keyword:\n\t\tarticle -> collect article information from zeit.de")
                print("\t\tcomment -> collect comment according to articles from zeit.de")
                print("\t\tusercomm -> collect comments from 50 users who wrote at least 100 different comments")
                print("\t\tpreprocess -> pre-processing of the collected user comments")
                print("\t\tfeature -> extract a set of features from the collected user comments)")

if __name__ == '__main__':
    do_work()
