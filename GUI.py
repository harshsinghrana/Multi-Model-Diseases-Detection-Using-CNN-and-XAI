import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk, ImageOps
import numpy as np
from keras.models import load_model

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Load the model and labels
try:
    model = load_model("hybrid_model_350.h5", compile=False)
except Exception as e:
    messagebox.showerror("Error", f"Failed to load model: {str(e)}")

class_names = [
    "Normal", "Normal No Cardiovascular Disease", 
    "Bacterial Pneumonia", "Tuberculosis",
    "Viral Pneumonia", "Cardiovascular Disease Found"
]

# Disease information dictionary
disease_info = {
    "Normal": {
        "description": "The individual has no noticeable signs of disease.",
        "precautions": "Maintain a healthy lifestyle, balanced diet, and regular exercise.",
        "medications": "None.",
        "diet": "Eat a balanced diet rich in vegetables, fruits, and whole grains."
    },
    "Normal No Cardiovascular Disease": {
        "description": "No cardiovascular disease detected, but should monitor overall health.",
        "precautions": "Maintain a healthy diet and regular physical activity.",
        "medications": "None.",
        "diet": "Consume a low-fat, low-sodium diet with plenty of fruits and vegetables."
    },
    "Bacterial Pneumonia": {
        "description": "A serious bacterial infection of the lungs that causes breathing difficulties.",
        "precautions": "Rest and hydration are essential. Avoid smoking and pollutants.",
        "medications": "Antibiotics like Amoxicillin, Azithromycin.",
        "diet": "Consume foods rich in Vitamin C and fluids like soups and teas."
    },
    "Tuberculosis": {
        "description": "A serious infectious disease that primarily affects the lungs.",
        "precautions": "Follow treatment regimens strictly. Avoid close contact with others during active stages.",
        "medications": "Isoniazid, Rifampin, Pyrazinamide.",
        "diet": "High-protein foods, iron-rich foods like leafy greens, lean meats."
    },
    "Viral Pneumonia": {
        "description": "Inflammation of the lungs caused by a viral infection, such as the flu.",
        "precautions": "Rest and stay hydrated. Isolate to avoid spreading the virus.",
        "medications": "Antiviral drugs if prescribed. Pain relievers like acetaminophen.",
        "diet": "Increase fluid intake, eat easily digestible food like soups."
    },
    "Cardiovascular Disease Found": {
        "description": "Heart-related conditions that affect the cardiovascular system.",
        "precautions": "Monitor blood pressure, cholesterol levels, and maintain a low-stress lifestyle.",
        "medications": "Aspirin, Beta-blockers, Statins, ACE inhibitors.",
        "diet": "Low-sodium, low-fat diet. High fiber from vegetables, fruits, and whole grains."
    }
}

# Configure styles
def configure_styles():
    style = ttk.Style()
    style.theme_use('clam')
    style.configure('Main.TFrame', background='#f0f0f0')
    style.configure('Title.TLabel', font=('Arial', 16, 'bold'), background='#3F51B5', foreground='white')
    style.configure('Section.TLabelframe', font=('Arial', 12, 'bold'), relief='groove', borderwidth=2)
    style.configure('Section.TLabelframe.Label', foreground='#3F51B5')
    style.configure('Result.TLabel', font=('Arial', 12, 'bold'), foreground='#4CAF50')
    style.configure('Custom.TButton', font=('Arial', 11, 'bold'), background='#4CAF50', foreground='white')
    style.map('Custom.TButton', background=[('active', '#45a049')])

class DiseaseClassifierApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Medical Imaging Diagnosis System")
        self.root.geometry("1200x900")
        self.root.configure(bg='#f0f0f0')
        
        configure_styles()
        self.create_widgets()
        
    def create_widgets(self):
        # Main container
        main_frame = ttk.Frame(self.root, style='Main.TFrame')
        main_frame.pack(fill='both', expand=True, padx=20, pady=20)

        # Title Section
        title_frame = ttk.Frame(main_frame)
        title_frame.pack(fill='x', pady=10)
        
        ttk.Label(title_frame, text="Medical Imaging Diagnosis System", style='Title.TLabel'
                 ).pack(side='left', padx=10, ipady=5)

        ttk.Button(title_frame, text="Upload Scan", command=self.select_image, 
                  style='Custom.TButton').pack(side='right', padx=10)

        # Image and Prediction Section
        top_frame = ttk.Frame(main_frame)
        top_frame.pack(fill='x', pady=10)

        # Image Preview
        self.img_frame = ttk.Labelframe(top_frame, text=" Scan Preview ", style='Section.TLabelframe')
        self.img_frame.pack(side='left', fill='both', expand=True, padx=5)
        
        self.input_img_label = ttk.Label(self.img_frame, background='white')
        self.input_img_label.pack(padx=10, pady=10)

        # Prediction Results
        result_frame = ttk.Labelframe(top_frame, text=" Diagnosis Results ", style='Section.TLabelframe')
        result_frame.pack(side='left', fill='both', expand=True, padx=5)
        
        self.result_label = ttk.Label(result_frame, text="Select an image to begin analysis", 
                                     style='Result.TLabel')
        self.result_label.pack(pady=5)
        
        self.confidence_label = ttk.Label(result_frame, text="", style='Result.TLabel')
        self.confidence_label.pack(pady=5)

        # Probabilities Table
        table_frame = ttk.Labelframe(main_frame, text=" Probability Distribution ", style='Section.TLabelframe')
        table_frame.pack(fill='both', expand=True, padx=5, pady=10)

        self.tree = ttk.Treeview(table_frame, columns=('Condition', 'Probability'), show='headings', height=6)
        self.tree.heading('Condition', text='Medical Condition')
        self.tree.heading('Probability', text='Probability (%)')
        self.tree.column('Condition', width=400)
        self.tree.column('Probability', width=200, anchor='center')
        self.tree.pack(fill='both', expand=True, padx=5, pady=5)

        # Disease Information Section
        info_frame = ttk.Frame(main_frame)
        info_frame.pack(fill='both', expand=True, pady=10)

        self.create_info_panel(info_frame, "Description", 0)
        self.create_info_panel(info_frame, "Precautions", 1)
        self.create_info_panel(info_frame, "Medications", 2)
        self.create_info_panel(info_frame, "Diet", 3)

    def create_info_panel(self, parent, title, column):
        frame = ttk.Labelframe(parent, text=f" {title} ", style='Section.TLabelframe')
        frame.grid(row=0, column=column, padx=5, pady=5, sticky='nsew')
        parent.grid_columnconfigure(column, weight=1)
        
        label = ttk.Label(frame, text="No information available", wraplength=250, justify='left')
        label.pack(fill='both', expand=True, padx=5, pady=5)
        setattr(self, f"{title.lower()}_label", label)

    def select_image(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            image = Image.open(file_path)
            self.classify_image(image)

    def classify_image(self, image):
        try:
            # Image preprocessing
            size = (224, 224)
            image = ImageOps.fit(image, size, Image.LANCZOS)
            if image.mode != "RGB": image = image.convert("RGB")
            
            # Display image
            input_img = ImageTk.PhotoImage(image)
            self.input_img_label.configure(image=input_img)
            self.input_img_label.image = input_img

            # Model prediction
            image_array = np.asarray(image)
            normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
            data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
            data[0] = normalized_image_array
            prediction = model.predict(data)
            
            # Process results
            index = np.argmax(prediction)
            probabilities = prediction[0]
            disease_name = class_names[index]
            confidence = probabilities[index]

            # Update UI components
            self.update_results(disease_name, confidence)
            self.update_probabilities(probabilities)
            self.update_disease_info(disease_name)

        except Exception as e:
            messagebox.showerror("Error", f"Classification error: {str(e)}")

    def update_results(self, disease, confidence):
        self.result_label.config(text=f"Primary Diagnosis: {disease}")
        self.confidence_label.config(text=f"Confidence Level: {confidence*100:.2f}%")

    def update_probabilities(self, probabilities):
        for i in self.tree.get_children():
            self.tree.delete(i)
            
        for i, prob in enumerate(probabilities):
            self.tree.insert('', 'end', values=(class_names[i], f"{prob*100:.2f}%"))

    def update_disease_info(self, disease):
        info = disease_info.get(disease, {})
        self.description_label.config(text=info.get("description", "Information not available"))
        self.precautions_label.config(text=info.get("precautions", "Information not available"))
        self.medications_label.config(text=info.get("medications", "Information not available"))
        self.diet_label.config(text=info.get("diet", "Information not available"))

if __name__ == "__main__":
    root = tk.Tk()
    app = DiseaseClassifierApp(root)
    root.mainloop()