import string
import tkinter as tk
from tkinter import filedialog
import tkinter.font as tkFont
import os
from nltk.corpus import stopwords
import LSA
import LDA4
import PLSA
import time


# Create a tkinter GUI window and a Text widget to display the file contents
root = tk.Tk()
root.geometry("900x820")
fontObj = tkFont.Font(size=14)
# Create a selector with various dummy options and pack it to the window
dummy_options = ["LDA", "LSA", "PLSA"]
method_selector = tk.StringVar(root)
method_selector.set(dummy_options[0])  # set the default option


def option_changed(*args):
    print(f"Selected option changed: {method_selector.get()}")


method_selector.trace("w", option_changed)  # trace changes to the selected option variable
dummy_selector = tk.OptionMenu(root, method_selector, *dummy_options)
dummy_selector.config(font=fontObj)
dummy_selector.pack()


dummy_options = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"]

no_of_topics = tk.StringVar(root)
no_of_topics.set(dummy_options[2])  # set the default option


def no_of_topics_changed(*args):
    print(f"Selected option changed: {no_of_topics.get()}")


no_of_topics.trace("w", no_of_topics_changed)  # trace changes to the selected option variable
dummy_selector = tk.OptionMenu(root, no_of_topics, *dummy_options)
dummy_selector.config(font=fontObj)
dummy_selector.pack()



# Create a Text widget to display the file contents and pack it to the window
text_widget = tk.Text(root, height=30, width=70
                      , font=fontObj)
text_widget.pack()

# Ask the user to select a folder and load all files in it
file_contents = []
tokenized_docs = []


def clear():
    root.update_idletasks()
    text_widget.delete("1.0", "end")
    text_widget.config()
    text_widget.insert(tk.END,
                       "Welcome to the topic modeler" +
                       "\n" + "please select the folder with texts files for topic "
                              "modelling")
    text_widget.config()
    tokenized_docs.clear()



clear()

def select_folder():
    folder_path = filedialog.askdirectory()
    file_contents = []
    for filename in os.listdir(folder_path):
        if os.path.isfile(os.path.join(folder_path, filename)):
            with open(os.path.join(folder_path, filename), 'r') as f:
                # file_contents.append(f.read())
                # text_widget.insert(tk.END, f"--- {filename} ---\n{file_contents[-1]}\n\n")
                r = f.read()
                r = r.translate(str.maketrans('', '', string.punctuation))  # removing punctuation

                file_token = r.split()
                file_token_lower = list(map(str.lower, file_token))  # converting the document into lowercase
                file_token_lower_stop_words = [word for word in file_token_lower if
                                               not word in stopwords.words()]
                tokenized_docs.append(file_token_lower_stop_words)


def execute():
    if method_selector.get() == "LDA":
        t1 = time.time()
        n = int(no_of_topics.get())
        string_to_be_printed = LDA4.LDA(tokenized_docs, n)
        text_widget.delete("1.0", "end")
        text_widget.config()
        # print(123)
        # print(string_to_be_printed)
        text_widget.insert(tk.END, string_to_be_printed)
        text_widget.config()
        t2 = time.time()
        print(t2-t1)
        # tokenized_docs.clear()

    if method_selector.get() == "LSA":
        t1 = time.time()
        n = int(no_of_topics.get())
        string_to_be_printed = LSA.LSA(tokenized_docs, n)
        text_widget.delete("1.0", "end")
        text_widget.config()
        # print(123)
        # print(string_to_be_printed)
        text_widget.insert(tk.END, string_to_be_printed)
        text_widget.config()
        t2 = time.time()
        print(t2 - t1)
        # tokenized_docs.clear()

    if method_selector.get() == "PLSA":
        t1 = time.time()
        n = int(no_of_topics.get())
        string_to_be_printed = PLSA.PLSA(tokenized_docs, n)
        text_widget.delete("1.0", "end")
        text_widget.config()
        # print(123)
        # print(string_to_be_printed)
        text_widget.insert(tk.END, string_to_be_printed)
        text_widget.config()
        t2 = time.time()
        print(t2 - t1)
        # tokenized_docs.clear()




select_button = tk.Button(root, text="Select Folder", command=select_folder, font=fontObj)
select_button.pack(side = tk.LEFT)
select_button.pack(padx=5)

execute_button = tk.Button(root, text="Execute", command=execute, font=fontObj)
execute_button.pack(side = tk.LEFT)
execute_button.pack(padx=5)

clear_button = tk.Button(root, text="Clear", command=clear, font=fontObj)
clear_button.pack(side = tk.LEFT)
clear_button.pack(padx=5)

# Set the Text widget to read-only mode


# Start the tkinter event loop
root.mainloop()

# Print the updated selected option after the event loop ends
print(f"Final selected option: {method_selector.get()}")
