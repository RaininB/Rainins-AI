
import tkinter as tk

def create_window():
    root = tk.Tk()
    label = tk.Label(root, text="Hello World")
    label.pack(padx=20, pady=20)
    root.mainloop()

create_window()
