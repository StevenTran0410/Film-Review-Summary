from Ultils import model_building, detect, model_word2vec
import warnings
warnings.filterwarnings("ignore")
import tkinter as tk
from collections import Counter

class App(tk.Tk):
    def __init__(self):
        super().__init__()

        self.title("Summary Review")

        self.number = 0
        self.selected_model = 1
        self.result = []
        self.model = model_building("Dense")
        self.word2vec_model = model_word2vec()

        self.model_options = ["Model 1 (DNN)", "Model 2 (RNN)", "Model 3 (Transformer)", "Model 4 (Hybrid)"]

        self.model_frame = tk.Frame(self, background=self.from_rgb((117, 123, 129)))
        self.model_frame.place(x=0, y=0, anchor="nw", width=800, height=75)

        select_model_button = tk.Button(self.model_frame, text="Select Model", command=self.change_model, font=("Arial", 12), width=12, relief=tk.RAISED, bg="#E8E8E8", activebackground="#D0D0D0")
        select_model_button.pack(side=tk.LEFT)

        self.model_label = tk.Label(self.model_frame, text=self.model_options[self.selected_model-1], font=("Arial", 12), width=20, relief=tk.SUNKEN, bg="#FFFFFF")
        self.model_label.pack(side=tk.LEFT, padx=10)

        summary_button = tk.Button(self.model_frame, text="Summary", command=self.show_summary, font=("Arial", 12), width=10, relief=tk.RAISED, bg="#E8E8E8", activebackground="#D0D0D0")
        summary_button.pack(side=tk.RIGHT)

        self.number_frame = tk.Frame(self)
        self.number_frame.place(x=300, y=75, anchor="nw", width=225, height=50)

        minus_button = tk.Button(self.number_frame, text="-", command=self.decrease_number, font=("Arial", 12), width=2, relief=tk.FLAT, bg="#E8E8E8", activebackground="#D0D0D0")
        minus_button.pack(side=tk.LEFT)

        self.number_field = tk.Entry(self.number_frame, font=("Arial", 12), width=10)
        self.number_field.insert(0, str(self.number))
        self.number_field.bind("<KeyRelease>", self.handle_number_input)
        self.number_field.pack(side=tk.LEFT, anchor="center")
        
        delete_all_button = tk.Button(self.number_frame, text="Delete All", command=self.delete_all_fields, font=("Arial", 12), width=7, relief=tk.FLAT, bg="#E8E8E8", activebackground="#D0D0D0")
        delete_all_button.pack(side=tk.RIGHT)
        
        plus_button = tk.Button(self.number_frame, text="+", command=self.increase_number, font=("Arial", 12), width=2, relief=tk.FLAT, bg="#E8E8E8", activebackground="#D0D0D0")
        plus_button.pack(side=tk.LEFT)

        self.insert_fields_frame = tk.Frame(self)
        self.insert_fields_frame.place(x=0, y=125, anchor="nw", width=800, height=600)

        self.show_summary_frame = tk.Frame(self, background=self.from_rgb((117, 123, 129)))
        self.show_summary_frame.place(x=0, y=655, anchor="nw", width=800, height=145)

        self.result_label = tk.Label(self.show_summary_frame, text="Result will be displayed here", font=("Arial", 12), wraplength=780, justify=tk.LEFT)
        self.result_label.pack()
        
        del_button = tk.Button(self.model_frame, text="Delete All Result", command=self.delete_result, font=("Arial", 12), width=13, relief=tk.RAISED, bg="#E8E8E8", activebackground="#D0D0D0")
        del_button.pack(side=tk.RIGHT)

        self.geometry("800x800") 

    def handle_number_input(self, event):
        try:
            new_number = int(self.number_field.get())
            if new_number < 0:                      # Limit the number to a minimum of 0 
                self.number = 0
                self.update_number_field()
                self.update_insert_fields()
            if new_number >= 0:
                self.number = min(new_number, 14)   # Limit the number to a maximum of 14
                self.update_number_field()
                self.update_insert_fields()
        except ValueError:
            pass

    def increase_number(self):
        self.number += 1
        if self.number > 14:
            self.number = 14
        self.update_number_field()
        self.update_insert_fields()

    def decrease_number(self):
        if self.number > 0:
            self.number -= 1
            self.update_number_field()
            self.update_insert_fields()

    def update_number_field(self):
        self.number_field.delete(0, tk.END)
        self.number_field.insert(0, str(self.number))

    def update_insert_fields(self):
        # Clear existing insert fields
        for widget in self.insert_fields_frame.winfo_children():
            widget.destroy()

        # Generate new insert fields
        for i in range(self.number):
            insert_field_frame = tk.Frame(self.insert_fields_frame)
            insert_field_frame.pack(pady=5)

            insert_field = tk.Entry(insert_field_frame, font=("Arial", 12), width=40)
            insert_field.pack(side=tk.LEFT)

            delete_button = tk.Button(insert_field_frame, text="Delete", command=lambda f=insert_field_frame: self.delete_field(f), font=("Arial", 10), width=8, relief=tk.RAISED, bg="#E8E8E8", activebackground="#D0D0D0")
            delete_button.pack(side=tk.LEFT, padx=5)

    def delete_field(self, field_frame):
        field_frame.destroy()
        self.number -= 1
        self.update_number_field()

    def delete_all_fields(self):
        for widget in self.insert_fields_frame.winfo_children():
            widget.destroy()
        self.number = 0
        self.update_number_field()

    def change_model(self):
        self.selected_model = (self.selected_model % len(self.model_options)) + 1
        self.model_label.config(text=self.model_options[self.selected_model-1])
        if self.selected_model == 1:
            self.model = model_building("Dense")
        elif self.selected_model == 2:
            pass
        elif self.selected_model == 3:
            pass
        elif self.selected_model == 4:
            pass

    def show_summary(self):
        self.process_text()
        total_reviews = len(self.result)
        if total_reviews == 0:
            summary_message = "Empty review"
        else:
            results_count = Counter(self.result)
            positive = results_count['Positive']
            negative = results_count['Negative']

            positive_percentage = (positive / total_reviews) * 100
            negative_percentage = (negative / total_reviews) * 100

            if positive_percentage > negative_percentage:
                recommendation = "recommend"
            else:
                recommendation = "not recommend"
            
            summary_message = f"There are {positive} positive reviews and {negative} negative reviews."
            summary_message += f" Positive reviews account for {positive_percentage:.2f}% of total reviews."
            if positive_percentage != negative_percentage:
                summary_message += f" Therefore, it is {recommendation} to watch this film/movie/anime."
            else:
                summary_message += f" Therefore, it is up to you to decide."

        self.result_label.config(text=summary_message)
            
    def delete_result(self):
        self.result = []
        self.result_label.config(text="Result will be displayed here")
                
    def from_rgb(self, rgb):
        return "#%02x%02x%02x" % rgb
    
    def process_text(self):
        for widget in self.insert_fields_frame.winfo_children():
            insert_field = widget.winfo_children()[0]
            text = insert_field.get()
            if text.strip():
                self.result.append(detect(text, self.model, self.word2vec_model))

        
app = App()
app.mainloop()