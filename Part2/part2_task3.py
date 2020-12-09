import sys
import subprocess

#create a function to process the argument/parameter
def do_work():
    args = sys.argv
    option = args[1:2]  # First element of args is the file name
    keyword = args[2:]

    # create the condition since argument/parameter is needed to get the result
    if len(option) == 0:
        print("You have not passed any commands in!\n\nPlease specify the parameter or use -h or -help to see the instruction.")

    # If the condition is met, it will process based on the argument/parameter
    else:
        for opt in option:
            #Print the instruction how to make use of the program
            if opt == '-h' or opt == '-help':
                print("This toolbox is for scraping article and comment information from the zeit.de which is a news webpage. \nMoreover, data from the webpage is able for preprocessing and generating features.")
                print("\n\nUsage: part2_task3.py [OPTIONS]\n")
                print("Options:\n\t-h\t\t\tprint overview information about this toolbox")
                print("\t-s [KEYWORD]\t\tstart toolbox in different operation modes")
                print("\nvalid keyword:\n\tarticle -> collect article information from zeit.de")
                print("\tcomment -> collect comment according to articles from zeit.de")
                print("\tusercomm -> collect comments from 50 users who wrote at least 100 different comments")
                print("\tpreprocess -> pre-processing of the collected user comments")
                print("\tfeature -> extract a set of features from the collected user comments")
            #if -s is chosen, keyword is needed to do the instruction
            elif opt == '-s':
                if keyword:
                    for key in keyword:
                        #call the file and run its code
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
                        #ask the user to put the valid keyword
                        else:
                            print("Please enter valid keyword")
                            print("valid keyword:\n\t\tarticle -> collect article information from zeit.de")
                            print("\t\tcomment -> collect comment according to articles from zeit.de")
                            print("\t\tusercomm -> collect comments from 50 users who wrote at least 100 different comments")
                            print("\t\tpreprocess -> pre-processing of the collected user comments")
                            print("\t\tfeature -> extract a set of features from the collected user comments")
                #if keyword is not matched, print the instruction again
                else:
                    print("Please enter valid keyword")
                    print("valid keyword:\n\t\tarticle -> collect article information from zeit.de")
                    print("\t\tcomment -> collect comment according to articles from zeit.de")
                    print("\t\tusercomm -> collect comments from 50 users who wrote at least 100 different comments")
                    print("\t\tpreprocess -> pre-processing of the collected user comments")
                    print("\t\tfeature -> extract a set of features from the collected user comments")
            # print this result if the argument or parameter is not matched
            else:
                print("Unrecognised argument.")
                print("Usage: part2_task3.py [OPTIONS]\n\n")
                print("Options:\n\t\t-h\t\t\t\tprint overview information about this toolbox")
                print("\t\t-s [KEYWORD]\t\t\tstart toolbox in different operation modes")
                print("valid keyword:\n\t\tarticle -> collect article information from zeit.de")
                print("\t\tcomment -> collect comment according to articles from zeit.de")
                print("\t\tusercomm -> collect comments from 50 users who wrote at least 100 different comments")
                print("\t\tpreprocess -> pre-processing of the collected user comments")
                print("\t\tfeature -> extract a set of features from the collected user comments")

#special variables
if __name__ == '__main__':
    do_work()
# do_work()