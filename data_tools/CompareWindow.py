

import tkinter as tk
from PIL import ImageTk, Image

class WindowComparison:
    def __init__(self):
        self.compare2int = {"<": -1, ">": 1, "=": 0}
        self.compare_val = None
        self.istarget = False

    def get_button_command(self, root, int_val):
        def close():
            root.destroy()
            self.compare_val = int_val
        return close

    def get_target_command(self, root):
        def set_is_target():
            root.destroy()
            self.istarget=True
        return set_is_target

    def open(self, title, path1, path2, t1, t2):
        window = tk.Tk()
        window_title = tk.Label(text=title)
        window_title.pack()


        compare_frame = tk.Frame(master=window, borderwidth=5)
        compare_frame.pack()
        f_image1 = tk.Frame(master=compare_frame, borderwidth=5)
        f_image1.pack(side=tk.LEFT)
        tk.Label(master=f_image1, text=t1).pack()
        img1 = ImageTk.PhotoImage(Image.open(path1).resize((250, 250), Image.ANTIALIAS))
        panel1 = tk.Label(f_image1, image=img1)
        panel1.pack()
        buttons_frame = tk.Frame(master=compare_frame, borderwidth=5)
        buttons_frame.pack(side=tk.LEFT)
        for str, i in self.compare2int.items():
            b = tk.Button(master=buttons_frame, text=str, command=self.get_button_command(window, i))
            b.pack()
        f_image2 = tk.Frame(master=compare_frame, borderwidth=5)
        f_image2.pack(side=tk.LEFT)
        tk.Label(master=f_image2, text=t2).pack()
        img2 = ImageTk.PhotoImage(Image.open(path2).resize((250, 250), Image.ANTIALIAS))
        panel2 = tk.Label(f_image2, image=img2)
        panel2.pack()
        target_b = tk.Button(master=f_image2, text="target", command=self.get_target_command(window))
        target_b.pack()
        window.mainloop()

# w = WindowComparison()
# w.open("which is more likely to stab?", "C:\\Users\\lotan\\Documents\\studies\\phoenix\\datasets\\256_ObjectCategories\\001.ak47\\001_0015.jpg",
#                 "C:\\Users\\lotan\\Documents\\studies\\phoenix\\datasets\\256_ObjectCategories\\001.ak47\\001_0015.jpg", "0", "1")
# print(w.compare_val)