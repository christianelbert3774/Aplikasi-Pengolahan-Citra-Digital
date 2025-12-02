import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import numpy as np
from datetime import datetime
import tkinter.simpledialog as simpledialog


class ImageProcessingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Processing App-Copyright © IMKTI-(2025)")
        self.root.geometry("1200x750")
        
        self.original_image = None
        self.processed_image = None
        self.current_image = None
        self.image_path = None
        self.stack_mode = tk.BooleanVar(value=False)  # Mode: False=Normal, True=Stack
        
        self.create_menu()
        self.create_canvas()
        self.create_control_buttons()

    def askinteger_slider(title, prompt, minvalue, maxvalue, initialvalue=0):
        popup = tk.Toplevel()
        popup.title(title)
        popup.geometry("300x150")
        popup.resizable(False, False)

        tk.Label(popup, text=prompt, font=("Arial", 11)).pack(pady=5)

        slider = tk.Scale(
            popup,
            from_=minvalue,
            to=maxvalue,
            orient=tk.HORIZONTAL,
            length=250
        )
        slider.set(initialvalue)
        slider.pack(pady=5)

        result = {"val": None}

        def confirm():
            result["val"] = slider.get()
            popup.destroy()

        tk.Button(popup, text="OK", command=confirm).pack(pady=5)

        popup.grab_set()
        popup.wait_window()

        return result["val"]
    
    def askfloat_slider(title, prompt, minvalue, maxvalue, initialvalue=0):
        popup = tk.Toplevel()
        popup.title(title)
        popup.geometry("300x150")
        popup.resizable(False, False)

        tk.Label(popup, text=prompt, font=("Arial", 11)).pack(pady=5)

        slider = tk.Scale(
            popup,
            from_=minvalue,
            to=maxvalue,
            orient=tk.HORIZONTAL,
            length=250
        )
        slider.set(initialvalue)
        slider.pack(pady=5)

        result = {"val": None}

        def confirm():
            result["val"] = slider.get()
            popup.destroy()

        tk.Button(popup, text="OK", command=confirm).pack(pady=5)

        popup.grab_set()
        popup.wait_window()

        return result["val"]

    simpledialog.askinteger = askinteger_slider
    simpledialog.askfloat = askfloat_slider
    
    def create_menu(self):
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)
        
        # File Menu
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Open...", command=self.open_image)
        file_menu.add_command(label="Save", command=self.save_image)
        file_menu.add_command(label="Save As", command=self.save_as_image)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)
        
        # Basic Ops Menu
        basic_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Basic Ops", menu=basic_menu)
        basic_menu.add_command(label="Negative", command=self.negative)
        basic_menu.add_command(label="Arithmetic", command=self.show_arithmetic_menu)
        basic_menu.add_command(label="Boolean", command=self.show_boolean_menu)
        basic_menu.add_command(label="Geometrics", command=self.show_geometrics_menu)
        basic_menu.add_command(label="Thresholding", command=self.thresholding)
        basic_menu.add_command(label="Convolution", command=self.show_convolution_menu)
        basic_menu.add_command(label="Fourier Transform", command=self.fourier_transform)
        basic_menu.add_command(label="Colouring", command=self.show_colouring_menu)
        
        # Enhancement Menu
        enhance_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Enhancement", menu=enhance_menu)
        enhance_menu.add_command(label="Brightness", command=self.brightness)
        enhance_menu.add_command(label="Contrast", command=self.contrast)
        enhance_menu.add_command(label="Hist. Equalization", command=self.hist_equalization)
        enhance_menu.add_command(label="Smoothing", command=self.show_smoothing_menu)
        enhance_menu.add_command(label="Sharpening", command=self.show_sharpening_menu)
        enhance_menu.add_command(label="Geometrics", command=self.show_geometrics_menu)
        enhance_menu.add_command(label="Correction", command=self.correction)
        
        # Noise Menu
        noise_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Noise", menu=noise_menu)
        noise_menu.add_command(label="Gaussian Noise", command=self.gaussian_noise)
        noise_menu.add_command(label="Rayleigh Noise", command=self.rayleigh_noise)
        noise_menu.add_command(label="Erlang (Gamma) Noise", command=self.gamma_noise)
        noise_menu.add_command(label="Exponential Noise", command=self.exponential_noise)
        noise_menu.add_command(label="Uniform Noise", command=self.uniform_noise)
        noise_menu.add_command(label="Impulse Noise", command=self.impulse_noise)
        
        # Edge Detection Menu
        edge_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Edge Detection", menu=edge_menu)
        
        edge_1st_menu = tk.Menu(edge_menu, tearoff=0)
        edge_menu.add_cascade(label="1st Differential Gradient", menu=edge_1st_menu)
        edge_1st_menu.add_command(label="Sobel", command=self.sobel)
        edge_1st_menu.add_command(label="Prewitt", command=self.prewitt)
        edge_1st_menu.add_command(label="Robert", command=self.robert)
        
        edge_2nd_menu = tk.Menu(edge_menu, tearoff=0)
        edge_menu.add_cascade(label="2nd Differential Gradient", menu=edge_2nd_menu)
        edge_2nd_menu.add_command(label="Laplacian", command=self.laplacian)
        edge_2nd_menu.add_command(label="Laplacian of Gaussian (LoG)", command=self.log)
        edge_2nd_menu.add_command(label="Canny", command=self.canny)
        
        edge_menu.add_command(label="Compass", command=self.show_compass_menu)
        
        # Segmentation Menu
        seg_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Segmentation", menu=seg_menu)
        seg_menu.add_command(label="Global Thresholding", command=self.global_thresholding)
        seg_menu.add_command(label="Region Growing", command=self.region_growing)
        seg_menu.add_command(label="Watershed", command=self.watershed)
        seg_menu.add_command(label="Edge Based Segmentation", command=self.edge_based_segmentation)
        
        # About Menu
        about_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="About", menu=about_menu)
        about_menu.add_command(label="Info Tim developer", command=self.show_info)
        about_menu.add_command(label="Tutorial", command=self.show_tutorial)
        
    def create_canvas(self):
    # Frame utama
        main_frame = tk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

    # Frame untuk gambar original
        left_frame = tk.Frame(main_frame)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)

        tk.Label(left_frame, text="Original Image", font=("Arial", 12, "bold")).pack()

    # Canvas adaptif (ikut melebar sesuai window)
        self.canvas_original = tk.Canvas(left_frame, bg="gray")
        self.canvas_original.pack(fill=tk.BOTH, expand=True)

    # Frame untuk hasil proses
        right_frame = tk.Frame(main_frame)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5)

        tk.Label(right_frame, text="Processed Image", font=("Arial", 12, "bold")).pack()

        self.canvas_processed = tk.Canvas(right_frame, bg="gray")
        self.canvas_processed.pack(fill=tk.BOTH, expand=True)

    
    def create_control_buttons(self):
        # Frame untuk tombol kontrol
        control_frame = tk.Frame(self.root)
        control_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # Frame kiri untuk mode selector
        left_controls = tk.Frame(control_frame)
        left_controls.pack(side=tk.LEFT, padx=5)
        
        # Mode Selector dengan visual yang jelas
        mode_frame = tk.LabelFrame(left_controls, text="Processing Mode", 
                                   font=("Arial", 9, "bold"), padx=10, pady=5)
        mode_frame.pack(side=tk.LEFT)
        
        self.mode_label = tk.Label(mode_frame, text="", font=("Arial", 9))
        self.mode_label.pack()
        
        mode_switch = tk.Checkbutton(mode_frame, text="Stack Mode (Auto-Apply)", 
                                    variable=self.stack_mode,
                                    command=self.toggle_mode,
                                    font=("Arial", 9, "bold"))
        mode_switch.pack()
        
        # Tombol Apply (hanya muncul di Normal Mode)
        self.btn_apply = tk.Button(control_frame, text="Apply to Original ➜", 
                                   command=self.apply_to_original, 
                                   bg="#4CAF50", fg="white", 
                                   font=("Arial", 10, "bold"),
                                   width=20, height=2)
        self.btn_apply.pack(side=tk.LEFT, padx=5)
        
        # Tombol Reset
        self.btn_reset = tk.Button(control_frame, text="⟲ Reset", 
                                   command=self.reset_image, 
                                   bg="#f44336", fg="white", 
                                   font=("Arial", 10, "bold"),
                                   width=15, height=2)
        self.btn_reset.pack(side=tk.LEFT, padx=5)
        
        # Label info
        self.info_label = tk.Label(control_frame, 
                                   text="", 
                                   font=("Arial", 9, "italic"),
                                   fg="blue")
        self.info_label.pack(side=tk.LEFT, padx=20)
        
        # Initialize mode display
        self.toggle_mode()
    
    def toggle_mode(self):
        """Toggle between Normal and Stack mode"""
        if self.stack_mode.get():
            # Stack Mode
            self.btn_apply.pack_forget()  # Hide Apply button
            self.mode_label.config(text="🔗 STACK MODE", fg="green")
            self.info_label.config(text="Stack Mode: Result shows first, auto-applies when you do NEXT operation", 
                                  fg="green")
        else:
            # Normal Mode
            self.btn_apply.pack(side=tk.LEFT, padx=5, after=self.btn_reset.master.winfo_children()[0])
            self.mode_label.config(text="🔓 NORMAL MODE", fg="blue")
            self.info_label.config(text="Normal Mode: Use 'Apply to Original' to stack effects", 
                                  fg="blue")
    
    def apply_to_original(self):
        """Apply processed image as new current image for stacking effects"""
        if self.processed_image is not None:
            self.current_image = self.processed_image.copy()
            self.display_image(self.current_image, self.canvas_original)
            # Clear processed canvas
            self.canvas_processed.delete("all")
            self.processed_image = None
            messagebox.showinfo("Success", "Processed image applied! You can now apply another effect.")
        else:
            messagebox.showwarning("Warning", "No processed image to apply!")
    
    def auto_apply_stack(self):
        """Automatically apply processed image in Stack Mode - called BEFORE next operation"""
        if self.processed_image is not None and self.stack_mode.get():
            self.current_image = self.processed_image.copy()
            self.display_image(self.current_image, self.canvas_original)
            # Clear processed canvas
            self.canvas_processed.delete("all")
            self.processed_image = None
    
    def prepare_for_operation(self):
        """Prepare for next operation in Stack Mode - auto apply previous result"""
        if self.stack_mode.get() and self.processed_image is not None:
            # Apply previous result as new current image
            self.current_image = self.processed_image.copy()
            self.display_image(self.current_image, self.canvas_original)
            # Clear processed canvas for new operation
            self.canvas_processed.delete("all")
            self.processed_image = None
    
    def reset_image(self):
        """Reset to original image"""
        if self.original_image is not None:
            self.current_image = self.original_image.copy()
            self.display_image(self.current_image, self.canvas_original)
            # Clear processed canvas
            self.canvas_processed.delete("all")
            self.processed_image = None
            messagebox.showinfo("Reset", "Image reset to original!")
        else:
            messagebox.showwarning("Warning", "No original image loaded!")
        
    def open_image(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff"), ("All files", "*.*")]
        )
        if file_path:
            self.image_path = file_path
            self.original_image = cv2.imread(file_path)
            self.current_image = self.original_image.copy()
            self.display_image(self.original_image, self.canvas_original)
            
    def save_image(self):
        if self.processed_image is not None:
            if self.image_path:
                cv2.imwrite(self.image_path, self.processed_image)
                messagebox.showinfo("Success", "Image saved successfully!")
            else:
                self.save_as_image()
        else:
            messagebox.showwarning("Warning", "No processed image to save!")
            
    def save_as_image(self):
        if self.processed_image is not None:
            file_path = filedialog.asksaveasfilename(
                defaultextension=".png",
                filetypes=[("PNG files", "*.png"), ("JPEG files", "*.jpg"), ("All files", "*.*")]
            )
            if file_path:
                cv2.imwrite(file_path, self.processed_image)
                messagebox.showinfo("Success", "Image saved successfully!")
        else:
            messagebox.showwarning("Warning", "No processed image to save!")
            
    def display_image(self, img, canvas, fit_to_canvas=True):
        if img is None:
            return
        
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img_rgb.shape[:2]

        # Dapatkan ukuran canvas *aktual*
        canvas.update()  # supaya canvas.winfo_width() sudah ter-update
        c_w = canvas.winfo_width()
        c_h = canvas.winfo_height()

        if fit_to_canvas:
            # Scaling agar gambar muat sempurna ke canvas
            scale = min(c_w / w, c_h / h)
            new_w = int(w * scale)
            new_h = int(h * scale)
            img_rgb = cv2.resize(img_rgb, (new_w, new_h))

        img_pil = Image.fromarray(img_rgb)
        img_tk = ImageTk.PhotoImage(img_pil)

        canvas.delete("all")

        # Tampilkan di tengah canvas
        canvas.create_image(c_w // 2, c_h // 2, image=img_tk, anchor="center")
        canvas.image = img_tk


        
    # ===== BASIC OPERATIONS =====
    def negative(self):
        if self.current_image is None:
            messagebox.showwarning("Warning", "Please open an image first!")
            return
        # Auto-apply previous result in Stack Mode before new operation
        self.prepare_for_operation()
        
        self.processed_image = 255 - self.current_image
        self.display_image(self.processed_image, self.canvas_processed)
        
    def show_arithmetic_menu(self):
        if self.current_image is None:
            messagebox.showwarning("Warning", "Please open an image first!")
            return
        
        menu = tk.Toplevel(self.root)
        menu.title("Arithmetic Operations")
        menu.geometry("300x200")
        
        tk.Button(menu, text="Add (+)", command=self.arithmetic_add, width=20).pack(pady=5)
        tk.Button(menu, text="Subtract (-)", command=self.arithmetic_subtract, width=20).pack(pady=5)
        tk.Button(menu, text="Multiply (*)", command=self.arithmetic_multiply, width=20).pack(pady=5)
        tk.Button(menu, text="Divide (/)", command=self.arithmetic_divide, width=20).pack(pady=5)
        
    def arithmetic_add(self):
        self.prepare_for_operation()
        value = tk.simpledialog.askinteger("Add", "Enter value to add (0-255):", minvalue=0, maxvalue=255)
        if value is not None:
            self.processed_image = cv2.add(self.current_image, np.array([value]))
            self.display_image(self.processed_image, self.canvas_processed)
            
    def arithmetic_subtract(self):
        self.prepare_for_operation()
        value = tk.simpledialog.askinteger("Subtract", "Enter value to subtract (0-255):", minvalue=0, maxvalue=255)
        if value is not None:
            self.processed_image = cv2.subtract(self.current_image, np.array([value]))
            self.display_image(self.processed_image, self.canvas_processed)
            
    def arithmetic_multiply(self):
        self.prepare_for_operation()
        value = tk.simpledialog.askfloat("Multiply", "Enter multiplier:", minvalue=0.1, maxvalue=5.0)
        if value is not None:
            self.processed_image = cv2.multiply(self.current_image, np.array([value]))
            self.processed_image = np.clip(self.processed_image, 0, 255).astype(np.uint8)
            self.display_image(self.processed_image, self.canvas_processed)
            
    def arithmetic_divide(self):
        self.prepare_for_operation()
        value = tk.simpledialog.askfloat("Divide", "Enter divisor:", minvalue=0.1, maxvalue=10.0)
        if value is not None:
            self.processed_image = cv2.divide(self.current_image, np.array([value]))
            self.processed_image = np.clip(self.processed_image, 0, 255).astype(np.uint8)
            self.display_image(self.processed_image, self.canvas_processed)
            
    def show_boolean_menu(self):
        if self.current_image is None:
            messagebox.showwarning("Warning", "Please open an image first!")
            return
            
        menu = tk.Toplevel(self.root)
        menu.title("Boolean Operations")
        menu.geometry("300x180")
        
        tk.Button(menu, text="NOT", command=self.boolean_not, width=20).pack(pady=5)
        tk.Button(menu, text="AND", command=self.boolean_and, width=20).pack(pady=5)
        tk.Button(menu, text="OR", command=self.boolean_or, width=20).pack(pady=5)
        tk.Button(menu, text="XOR", command=self.boolean_xor, width=20).pack(pady=5)
        
    def boolean_not(self):
        self.prepare_for_operation()
        self.processed_image = cv2.bitwise_not(self.current_image)
        self.display_image(self.processed_image, self.canvas_processed)
        
    def boolean_and(self):
        self.prepare_for_operation()
        mask = np.ones_like(self.current_image) * 128
        self.processed_image = cv2.bitwise_and(self.current_image, mask)
        self.display_image(self.processed_image, self.canvas_processed)
        
    def boolean_or(self):
        self.prepare_for_operation()
        mask = np.ones_like(self.current_image) * 128
        self.processed_image = cv2.bitwise_or(self.current_image, mask)
        self.display_image(self.processed_image, self.canvas_processed)
        
    def boolean_xor(self):
        self.prepare_for_operation()
        mask = np.ones_like(self.current_image) * 128
        self.processed_image = cv2.bitwise_xor(self.current_image, mask)
        self.display_image(self.processed_image, self.canvas_processed)
        
    def show_geometrics_menu(self):
        if self.current_image is None:
            messagebox.showwarning("Warning", "Please open an image first!")
            return
            
        menu = tk.Toplevel(self.root)
        menu.title("Geometric Operations")
        menu.geometry("300x200")
        
        tk.Button(menu, text="Translation", command=self.translation, width=20).pack(pady=5)
        tk.Button(menu, text="Rotation", command=self.rotation, width=20).pack(pady=5)
        tk.Button(menu, text="Zooming", command=self.zooming, width=20).pack(pady=5)
        tk.Button(menu, text="Flipping", command=self.flipping, width=20).pack(pady=5)
        tk.Button(menu, text="Cropping", command=self.cropping, width=20).pack(pady=5)
        
    def translation(self):
        self.prepare_for_operation()

    # Input translasi X dan Y dengan batas aman
        tx = tk.simpledialog.askinteger(
            "Translation",
            "Enter X translation:",
            initialvalue=0,
            minvalue=-1000,
            maxvalue=1000
        )
        ty = tk.simpledialog.askinteger(
            "Translation",
            "Enter Y translation:",
            initialvalue=0,
            minvalue=-1000,
            maxvalue=1000
        )

        if tx is not None and ty is not None:
            rows, cols = self.current_image.shape[:2]
            M = np.float32([[1, 0, tx], [0, 1, ty]])
            self.processed_image = cv2.warpAffine(self.current_image, M, (cols, rows))
            self.display_image(self.processed_image, self.canvas_processed)


    def rotation(self):
        self.prepare_for_operation()

        angle = tk.simpledialog.askfloat(
            "Rotation",
            "Enter rotation angle (degrees):",
            initialvalue=0,
            minvalue=-360,
            maxvalue=360
        )

        if angle is not None:
            rows, cols = self.current_image.shape[:2]
            M = cv2.getRotationMatrix2D((cols/2, rows/2), angle, 1)
            self.processed_image = cv2.warpAffine(self.current_image, M, (cols, rows))
            self.display_image(self.processed_image, self.canvas_processed)


    def zooming(self):
        self.prepare_for_operation()

    # Pastikan gambar ada
        if self.current_image is None:
            messagebox.showerror("Error", "Tidak ada gambar yang sedang ditampilkan.")
            return

    # Input faktor zoom
        scale = tk.simpledialog.askfloat(
            "Zooming",
            "Masukkan faktor zoom (1 - 5):",
            minvalue=1.0,
            maxvalue=5.0,
            initialvalue=0.0
        )
        if scale is None:
            return  # User cancel

    # Hitung ukuran baru
        h, w = self.current_image.shape[:2]
        new_h, new_w = int(h * scale), int(w * scale)

    # Antisipasi crash jika hasil resize 0
        new_h = max(1, new_h)
        new_w = max(1, new_w)

    # Proses resize / zoom
        zoomed_img = cv2.resize(self.current_image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        self.processed_image = zoomed_img

    # Tampilkan tanpa fit-to-canvas
        self.display_image(self.processed_image, self.canvas_processed, fit_to_canvas=False)
        self.canvas_processed.update()


            
    def flipping(self):
        flip_menu = tk.Toplevel(self.root)
        flip_menu.title("Flip Direction")
        flip_menu.geometry("250x150")
        
        tk.Button(flip_menu, text="Horizontal", command=lambda: self.do_flip(1), width=15).pack(pady=5)
        tk.Button(flip_menu, text="Vertical", command=lambda: self.do_flip(0), width=15).pack(pady=5)
        tk.Button(flip_menu, text="Both", command=lambda: self.do_flip(-1), width=15).pack(pady=5)
        
    def do_flip(self, flip_code):
        self.prepare_for_operation()
        self.processed_image = cv2.flip(self.current_image, flip_code)
        self.display_image(self.processed_image, self.canvas_processed)
        
    def cropping(self):
        self.prepare_for_operation()
        # Manual cropping dengan input persentase
        crop_percent = tk.simpledialog.askinteger(
            "Cropping", 
            "Masukkan nilai crop (10-40%):", 
            minvalue=10, 
            maxvalue=40, 
            initialvalue=25
        )
        if crop_percent is not None:
            h, w = self.current_image.shape[:2]
            crop_ratio = crop_percent / 100.0
            
            x1 = int(w * crop_ratio)
            y1 = int(h * crop_ratio)
            x2 = int(w * (1 - crop_ratio))
            y2 = int(h * (1 - crop_ratio))
            
            self.processed_image = self.current_image[y1:y2, x1:x2]
            self.display_image(self.processed_image, self.canvas_processed)
        
    def thresholding(self):
        if self.current_image is None:
            messagebox.showwarning("Warning", "Please open an image first!")
            return
        self.prepare_for_operation()
        
        gray = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2GRAY)
        _, self.processed_image = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        self.processed_image = cv2.cvtColor(self.processed_image, cv2.COLOR_GRAY2BGR)
        self.display_image(self.processed_image, self.canvas_processed)
    
    def show_convolution_menu(self):
        if self.current_image is None:
            messagebox.showwarning("Warning", "Please open an image first!")
            return
            
        menu = tk.Toplevel(self.root)
        menu.title("Convolution Filters")
        menu.geometry("350x300")
        
        tk.Label(menu, text="Select Convolution Kernel:", font=("Arial", 10, "bold")).pack(pady=10)
        
        tk.Button(menu, text="Identity", command=lambda: self.apply_convolution("identity"), width=25).pack(pady=3)
        tk.Button(menu, text="Edge Detection", command=lambda: self.apply_convolution("edge"), width=25).pack(pady=3)
        tk.Button(menu, text="Sharpen", command=lambda: self.apply_convolution("sharpen"), width=25).pack(pady=3)
        tk.Button(menu, text="Box Blur", command=lambda: self.apply_convolution("box_blur"), width=25).pack(pady=3)
        tk.Button(menu, text="Gaussian Blur", command=lambda: self.apply_convolution("gaussian"), width=25).pack(pady=3)
        tk.Button(menu, text="Emboss", command=lambda: self.apply_convolution("emboss"), width=25).pack(pady=3)
        tk.Button(menu, text="Custom Kernel", command=self.custom_convolution, width=25).pack(pady=3)
        
    def apply_convolution(self, kernel_type):
        self.prepare_for_operation()
        if kernel_type == "identity":
            kernel = np.array([[0, 0, 0],
                              [0, 1, 0],
                              [0, 0, 0]], dtype=np.float32)
        elif kernel_type == "edge":
            kernel = np.array([[-1, -1, -1],
                              [-1,  8, -1],
                              [-1, -1, -1]], dtype=np.float32)
        elif kernel_type == "sharpen":
            kernel = np.array([[0, -1, 0],
                              [-1, 5, -1],
                              [0, -1, 0]], dtype=np.float32)
        elif kernel_type == "box_blur":
            kernel = np.ones((9, 9), dtype=np.float32) / 81.0
        elif kernel_type == "gaussian":
            kernel = np.array([[1, 4, 7, 4, 1],
                              [4, 16, 26, 16, 4],
                              [7, 26, 41, 26, 7],
                              [4, 16, 26, 16, 4],
                              [1, 4, 7, 4, 1]], dtype=np.float32) / 273.0
        elif kernel_type == "emboss":
            kernel = np.array([[-2, -1, 0],
                              [-1,  1, 1],
                              [ 0,  1, 2]], dtype=np.float32)
        
        self.processed_image = cv2.filter2D(self.current_image, -1, kernel)
        
        if kernel_type == "emboss":
            self.processed_image = cv2.convertScaleAbs(self.processed_image, alpha=1, beta=128)
        
        self.display_image(self.processed_image, self.canvas_processed)
        
    def custom_convolution(self):
        self.prepare_for_operation()
        custom_window = tk.Toplevel(self.root)
        custom_window.title("Custom Convolution Kernel")
        custom_window.geometry("350x400")
        
        tk.Label(custom_window, text="Enter 3x3 Kernel Values:", font=("Arial", 10, "bold")).pack(pady=10)
        
        entries = []
        frame = tk.Frame(custom_window)
        frame.pack(pady=10)
        
        for i in range(3):
            row_entries = []
            for j in range(3):
                entry = tk.Entry(frame, width=8)
                entry.grid(row=i, column=j, padx=5, pady=5)
                entry.insert(0, "0" if i != 1 or j != 1 else "1")
                row_entries.append(entry)
            entries.append(row_entries)
        
        def apply_custom():
            try:
                kernel = np.array([[float(entries[i][j].get()) for j in range(3)] for i in range(3)])
                self.processed_image = cv2.filter2D(self.current_image, -1, kernel)
                self.display_image(self.processed_image, self.canvas_processed)
                custom_window.destroy()
                messagebox.showinfo("Success", "Custom convolution applied!")
            except ValueError:
                messagebox.showerror("Error", "Please enter valid numbers!")
        
        tk.Button(custom_window, text="Apply Convolution", command=apply_custom, width=20).pack(pady=10)
        
    def fourier_transform(self):
        if self.current_image is None:
            messagebox.showwarning("Warning", "Please open an image first!")
            return
        self.prepare_for_operation()
            
        gray = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2GRAY)
        f = np.fft.fft2(gray)
        fshift = np.fft.fftshift(f)
        magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1)
        
        magnitude_spectrum = cv2.normalize(magnitude_spectrum, None, 0, 255, cv2.NORM_MINMAX)
        self.processed_image = cv2.cvtColor(magnitude_spectrum.astype(np.uint8), cv2.COLOR_GRAY2BGR)
        self.display_image(self.processed_image, self.canvas_processed)
        
    def show_colouring_menu(self):
        if self.current_image is None:
            messagebox.showwarning("Warning", "Please open an image first!")
            return
            
        menu = tk.Toplevel(self.root)
        menu.title("Colouring")
        menu.geometry("300x250")
        
        tk.Button(menu, text="Binary", command=self.binary, width=20).pack(pady=3)
        tk.Button(menu, text="Grayscale", command=self.grayscale, width=20).pack(pady=3)
        tk.Button(menu, text="RGB", command=self.rgb, width=20).pack(pady=3)
        tk.Button(menu, text="HSV", command=self.hsv, width=20).pack(pady=3)
        tk.Button(menu, text="CMY", command=self.cmy, width=20).pack(pady=3)
        tk.Button(menu, text="YUV", command=self.yuv, width=20).pack(pady=3)
        tk.Button(menu, text="YIQ", command=self.yiq, width=20).pack(pady=3)
        tk.Button(menu, text="Pseudo", command=self.pseudo, width=20).pack(pady=3)
        
    def binary(self):
        self.prepare_for_operation()
        gray = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        self.processed_image = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
        self.display_image(self.processed_image, self.canvas_processed)
        
    def grayscale(self):
        self.prepare_for_operation()
        gray = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2GRAY)
        self.processed_image = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        self.display_image(self.processed_image, self.canvas_processed)
        
    def rgb(self):
        self.prepare_for_operation()
        self.processed_image = self.current_image.copy()
        self.display_image(self.processed_image, self.canvas_processed)
        
    def hsv(self):
        self.prepare_for_operation()
        hsv = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2HSV)
        self.processed_image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        self.display_image(self.processed_image, self.canvas_processed)
        
    def cmy(self):
        self.prepare_for_operation()
        cmy = 255 - self.current_image
        self.processed_image = cmy
        self.display_image(self.processed_image, self.canvas_processed)
        
    def yuv(self):
        self.prepare_for_operation()
        yuv = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2YUV)
        self.processed_image = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
        self.display_image(self.processed_image, self.canvas_processed)
        
    def yiq(self):
        self.prepare_for_operation()
        # YIQ conversion
        transform_matrix = np.array([[0.299, 0.587, 0.114],
                                     [0.596, -0.275, -0.321],
                                     [0.212, -0.523, 0.311]])
        img_float = self.current_image.astype(float) / 255.0
        yiq = np.dot(img_float, transform_matrix.T)
        yiq = np.clip(yiq, 0, 1)
        self.processed_image = (yiq * 255).astype(np.uint8)
        self.display_image(self.processed_image, self.canvas_processed)
        
    def pseudo(self):
        self.prepare_for_operation()
        gray = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2GRAY)
        self.processed_image = cv2.applyColorMap(gray, cv2.COLORMAP_JET)
        self.display_image(self.processed_image, self.canvas_processed)
        
    # ===== ENHANCEMENT =====
    def brightness(self):
        self.prepare_for_operation()
        value = tk.simpledialog.askinteger("Brightness", "Enter brightness value (-100 to 100):", 
                                          minvalue=-100, maxvalue=100, initialvalue=30)
        if value is not None:
            self.processed_image = cv2.convertScaleAbs(self.current_image, alpha=1, beta=value)
            self.display_image(self.processed_image, self.canvas_processed)
            
    def contrast(self):
        self.prepare_for_operation()
        value = tk.simpledialog.askfloat("Contrast", "Enter contrast factor (0.5 to 3.0):", 
                                        minvalue=0.5, maxvalue=3.0, initialvalue=1.5)
        if value is not None:
            self.processed_image = cv2.convertScaleAbs(self.current_image, alpha=value, beta=0)
            self.display_image(self.processed_image, self.canvas_processed)
            
    def hist_equalization(self):
        self.prepare_for_operation()
        if len(self.current_image.shape) == 3:
            img_yuv = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2YUV)
            img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
            self.processed_image = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
        else:
            self.processed_image = cv2.equalizeHist(self.current_image)
        self.display_image(self.processed_image, self.canvas_processed)
        
    def show_smoothing_menu(self):
        if self.current_image is None:
            messagebox.showwarning("Warning", "Please open an image first!")
            return
            
        menu = tk.Toplevel(self.root)
        menu.title("Smoothing Filters")
        menu.geometry("300x200")
        
        tk.Label(menu, text="Spatial Domain:", font=("Arial", 10, "bold")).pack(pady=5)
        tk.Button(menu, text="Lowpass Filtering", command=self.lowpass_filter, width=25).pack(pady=2)
        tk.Button(menu, text="Median Filtering", command=self.median_filter, width=25).pack(pady=2)
        
        tk.Label(menu, text="Frequency Domain:", font=("Arial", 10, "bold")).pack(pady=5)
        tk.Button(menu, text="ILPF", command=self.ilpf, width=25).pack(pady=2)
        tk.Button(menu, text="BLPF", command=self.blpf, width=25).pack(pady=2)
        
    def lowpass_filter(self):
        self.prepare_for_operation()
        kernel_size = tk.simpledialog.askinteger("Lowpass Filter", "Enter kernel size (odd number):", 
                                                 minvalue=3, maxvalue=15, initialvalue=5)
        if kernel_size and kernel_size % 2 == 1:
            self.processed_image = cv2.blur(self.current_image, (kernel_size, kernel_size))
            self.display_image(self.processed_image, self.canvas_processed)
            
    def median_filter(self):
        self.prepare_for_operation()
        kernel_size = tk.simpledialog.askinteger("Median Filter", "Enter kernel size (odd number):", 
                                                 minvalue=3, maxvalue=15, initialvalue=5)
        if kernel_size and kernel_size % 2 == 1:
            self.processed_image = cv2.medianBlur(self.current_image, kernel_size)
            self.display_image(self.processed_image, self.canvas_processed)
            
    def ilpf(self):
        self.prepare_for_operation()
        gray = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2GRAY)
        
        radius = tk.simpledialog.askinteger("ILPF", "Enter cutoff radius (10-100):", 
                                           minvalue=10, maxvalue=100, initialvalue=30)
        if radius is None:
            return
        
        dft = cv2.dft(np.float32(gray), flags=cv2.DFT_COMPLEX_OUTPUT)
        dft_shift = np.fft.fftshift(dft)
        
        rows, cols = gray.shape
        crow, ccol = rows//2, cols//2
        
        mask = np.zeros((rows, cols, 2), np.float32)
        for u in range(rows):
            for v in range(cols):
                d = np.sqrt((u - crow)**2 + (v - ccol)**2)
                if d <= radius:
                    mask[u, v] = 1
        
        fshift = dft_shift * mask
        f_ishift = np.fft.ifftshift(fshift)
        img_back = cv2.idft(f_ishift)
        img_back = cv2.magnitude(img_back[:,:,0], img_back[:,:,1])
        
        self.processed_image = cv2.normalize(img_back, None, 0, 255, cv2.NORM_MINMAX)
        self.processed_image = cv2.cvtColor(self.processed_image.astype(np.uint8), cv2.COLOR_GRAY2BGR)
        self.display_image(self.processed_image, self.canvas_processed)
        
    def blpf(self):
        self.prepare_for_operation()
        gray = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2GRAY)
        
        radius = tk.simpledialog.askinteger("BLPF", "Enter cutoff radius (10-100):", 
                                           minvalue=10, maxvalue=100, initialvalue=30)
        if radius is None:
            return
            
        order = tk.simpledialog.askinteger("BLPF", "Enter filter order (1-5):", 
                                          minvalue=1, maxvalue=5, initialvalue=2)
        if order is None:
            return
        
        dft = cv2.dft(np.float32(gray), flags=cv2.DFT_COMPLEX_OUTPUT)
        dft_shift = np.fft.fftshift(dft)
        
        rows, cols = gray.shape
        crow, ccol = rows//2, cols//2
        
        mask = np.zeros((rows, cols, 2), np.float32)
        for u in range(rows):
            for v in range(cols):
                d = np.sqrt((u - crow)**2 + (v - ccol)**2)
                mask[u, v] = 1 / (1 + (d / radius)**(2 * order))
        
        fshift = dft_shift * mask
        f_ishift = np.fft.ifftshift(fshift)
        img_back = cv2.idft(f_ishift)
        img_back = cv2.magnitude(img_back[:,:,0], img_back[:,:,1])
        
        self.processed_image = cv2.normalize(img_back, None, 0, 255, cv2.NORM_MINMAX)
        self.processed_image = cv2.cvtColor(self.processed_image.astype(np.uint8), cv2.COLOR_GRAY2BGR)
        self.display_image(self.processed_image, self.canvas_processed)
        
    def show_sharpening_menu(self):
        if self.current_image is None:
            messagebox.showwarning("Warning", "Please open an image first!")
            return
            
        menu = tk.Toplevel(self.root)
        menu.title("Sharpening Filters")
        menu.geometry("300x180")
        
        tk.Label(menu, text="Spatial Domain:", font=("Arial", 10, "bold")).pack(pady=5)
        tk.Button(menu, text="Highpass Filtering", command=self.highpass_filter, width=25).pack(pady=2)
        tk.Button(menu, text="Highboost Filtering", command=self.highboost_filter, width=25).pack(pady=2)
        
        tk.Label(menu, text="Frequency Domain:", font=("Arial", 10, "bold")).pack(pady=5)
        tk.Button(menu, text="IHPF", command=self.ihpf, width=25).pack(pady=2)
        tk.Button(menu, text="BHPF", command=self.bhpf, width=25).pack(pady=2)
        
    def highpass_filter(self):
        self.prepare_for_operation()
        kernel = np.array([[-1, -1, -1],
                          [-1,  8, -1],
                          [-1, -1, -1]])
        self.processed_image = cv2.filter2D(self.current_image, -1, kernel)
        self.display_image(self.processed_image, self.canvas_processed)
        
    def highboost_filter(self):
        self.prepare_for_operation()
        k = tk.simpledialog.askfloat("Highboost", "Enter boost factor:", 
                                     minvalue=1.0, maxvalue=5.0, initialvalue=1.5)
        if k is not None:
            blurred = cv2.GaussianBlur(self.current_image, (5, 5), 0)
            self.processed_image = cv2.addWeighted(self.current_image, k, blurred, 1-k, 0)
            self.display_image(self.processed_image, self.canvas_processed)
            
    def ihpf(self):
        self.prepare_for_operation()
        gray = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2GRAY)
        
        radius = tk.simpledialog.askinteger("IHPF", "Enter cutoff radius (10-100):", 
                                           minvalue=10, maxvalue=100, initialvalue=30)
        if radius is None:
            return
        
        dft = cv2.dft(np.float32(gray), flags=cv2.DFT_COMPLEX_OUTPUT)
        dft_shift = np.fft.fftshift(dft)
        
        rows, cols = gray.shape
        crow, ccol = rows//2, cols//2
        
        mask = np.ones((rows, cols, 2), np.float32)
        for u in range(rows):
            for v in range(cols):
                d = np.sqrt((u - crow)**2 + (v - ccol)**2)
                if d <= radius:
                    mask[u, v] = 0
        
        fshift = dft_shift * mask
        f_ishift = np.fft.ifftshift(fshift)
        img_back = cv2.idft(f_ishift)
        img_back = cv2.magnitude(img_back[:,:,0], img_back[:,:,1])
        
        self.processed_image = cv2.normalize(img_back, None, 0, 255, cv2.NORM_MINMAX)
        self.processed_image = cv2.cvtColor(self.processed_image.astype(np.uint8), cv2.COLOR_GRAY2BGR)
        self.display_image(self.processed_image, self.canvas_processed)
        
    def bhpf(self):
        self.prepare_for_operation()
        gray = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2GRAY)
        
        radius = tk.simpledialog.askinteger("BHPF", "Enter cutoff radius (10-100):", 
                                           minvalue=10, maxvalue=100, initialvalue=30)
        if radius is None:
            return
            
        order = tk.simpledialog.askinteger("BHPF", "Enter filter order (1-5):", 
                                          minvalue=1, maxvalue=5, initialvalue=2)
        if order is None:
            return
        
        dft = cv2.dft(np.float32(gray), flags=cv2.DFT_COMPLEX_OUTPUT)
        dft_shift = np.fft.fftshift(dft)
        
        rows, cols = gray.shape
        crow, ccol = rows//2, cols//2
        
        mask = np.zeros((rows, cols, 2), np.float32)
        for u in range(rows):
            for v in range(cols):
                d = np.sqrt((u - crow)**2 + (v - ccol)**2)
                if d == 0:
                    mask[u, v] = 0
                else:
                    mask[u, v] = 1 / (1 + (radius / d)**(2 * order))
        
        fshift = dft_shift * mask
        f_ishift = np.fft.ifftshift(fshift)
        img_back = cv2.idft(f_ishift)
        img_back = cv2.magnitude(img_back[:,:,0], img_back[:,:,1])
        
        self.processed_image = cv2.normalize(img_back, None, 0, 255, cv2.NORM_MINMAX)
        self.processed_image = cv2.cvtColor(self.processed_image.astype(np.uint8), cv2.COLOR_GRAY2BGR)
        self.display_image(self.processed_image, self.canvas_processed)
        
    def correction(self):
        if self.current_image is None:
            messagebox.showwarning("Warning", "Please open an image first!")
            return
        self.prepare_for_operation()
        
        gamma = tk.simpledialog.askfloat("Gamma Correction", "Enter gamma value:", 
                                        minvalue=0.1, maxvalue=3.0, initialvalue=1.5)
        if gamma is not None:
            invGamma = 1.0 / gamma
            table = np.array([((i / 255.0) ** invGamma) * 255 
                            for i in np.arange(0, 256)]).astype("uint8")
            self.processed_image = cv2.LUT(self.current_image, table)
            self.display_image(self.processed_image, self.canvas_processed)
            
    # ===== NOISE =====
    def gaussian_noise(self):
        self.prepare_for_operation()
        mean = 0
        sigma = tk.simpledialog.askfloat("Gaussian Noise", "Enter sigma value:", 
                                        minvalue=1, maxvalue=50, initialvalue=25)
        if sigma is not None:
            gauss = np.random.normal(mean, sigma, self.current_image.shape).astype('uint8')
            self.processed_image = cv2.add(self.current_image, gauss)
            self.display_image(self.processed_image, self.canvas_processed)
            
    def rayleigh_noise(self):
        self.prepare_for_operation()
        scale = tk.simpledialog.askfloat("Rayleigh Noise", "Enter scale value:", 
                                        minvalue=1, maxvalue=50, initialvalue=20)
        if scale is not None:
            noise = np.random.rayleigh(scale, self.current_image.shape).astype('uint8')
            self.processed_image = cv2.add(self.current_image, noise)
            self.display_image(self.processed_image, self.canvas_processed)
            
    def gamma_noise(self):
        self.prepare_for_operation()
        shape = tk.simpledialog.askfloat("Gamma Noise", "Enter shape parameter:", 
                                        minvalue=1, maxvalue=10, initialvalue=2)
        if shape is not None:
            noise = np.random.gamma(shape, 2.0, self.current_image.shape).astype('uint8')
            self.processed_image = cv2.add(self.current_image, noise)
            self.display_image(self.processed_image, self.canvas_processed)
            
    def exponential_noise(self):
        self.prepare_for_operation()
        scale = tk.simpledialog.askfloat("Exponential Noise", "Enter scale value:", 
                                        minvalue=1, maxvalue=50, initialvalue=20)
        if scale is not None:
            noise = np.random.exponential(scale, self.current_image.shape).astype('uint8')
            self.processed_image = cv2.add(self.current_image, noise)
            self.display_image(self.processed_image, self.canvas_processed)
            
    def uniform_noise(self):
        self.prepare_for_operation()
        low = tk.simpledialog.askinteger("Uniform Noise", "Enter low value:", 
                                        minvalue=0, maxvalue=100, initialvalue=0)
        high = tk.simpledialog.askinteger("Uniform Noise", "Enter high value:", 
                                         minvalue=0, maxvalue=100, initialvalue=50)
        if low is not None and high is not None:
            noise = np.random.uniform(low, high, self.current_image.shape).astype('uint8')
            self.processed_image = cv2.add(self.current_image, noise)
            self.display_image(self.processed_image, self.canvas_processed)
            
    def impulse_noise(self):
        self.prepare_for_operation()
        prob = tk.simpledialog.askfloat("Impulse Noise", "Enter probability (0.01-0.1):", 
                                       minvalue=0.01, maxvalue=0.1, initialvalue=0.05)
        if prob is not None:
            self.processed_image = self.current_image.copy()
            num_salt = np.ceil(prob * self.current_image.size * 0.5)
            coords = [np.random.randint(0, i - 1, int(num_salt)) 
                     for i in self.current_image.shape]
            self.processed_image[coords[0], coords[1], :] = 255
            
            num_pepper = np.ceil(prob * self.current_image.size * 0.5)
            coords = [np.random.randint(0, i - 1, int(num_pepper)) 
                     for i in self.current_image.shape]
            self.processed_image[coords[0], coords[1], :] = 0
            
            self.display_image(self.processed_image, self.canvas_processed)
            
    # ===== EDGE DETECTION =====
    def sobel(self):
        self.prepare_for_operation()
        gray = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2GRAY)

        sobelx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)

        sobel = cv2.magnitude(sobelx, sobely)
        sobel = cv2.normalize(sobel, None, 0, 255, cv2.NORM_MINMAX)

        self.processed_image = cv2.cvtColor(sobel.astype(np.uint8), cv2.COLOR_GRAY2BGR)
        self.display_image(self.processed_image, self.canvas_processed)


    def prewitt(self):
        self.prepare_for_operation()
        gray = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2GRAY)

        kernelx = np.array([[1, 0, -1],
                        [1, 0, -1],
                        [1, 0, -1]], dtype=np.float32)

        kernely = np.array([[1,  1,  1],
                        [0,  0,  0],
                        [-1, -1, -1]], dtype=np.float32)

        prewittx = cv2.filter2D(gray, cv2.CV_32F, kernelx)
        prewitty = cv2.filter2D(gray, cv2.CV_32F, kernely)

        prewitt = cv2.magnitude(prewittx, prewitty)
        prewitt = cv2.normalize(prewitt, None, 0, 255, cv2.NORM_MINMAX)

        self.processed_image = cv2.cvtColor(prewitt.astype(np.uint8), cv2.COLOR_GRAY2BGR)
        self.display_image(self.processed_image, self.canvas_processed)


    def robert(self):
        self.prepare_for_operation()
        gray = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2GRAY)

        kernelx = np.array([[1, 0],
                        [0, -1]], dtype=np.float32)

        kernely = np.array([[0, 1],
                        [-1, 0]], dtype=np.float32)

        robertx = cv2.filter2D(gray, cv2.CV_32F, kernelx)
        roberty = cv2.filter2D(gray, cv2.CV_32F, kernely)

        robert = cv2.magnitude(robertx, roberty)
        robert = cv2.normalize(robert, None, 0, 255, cv2.NORM_MINMAX)

        self.processed_image = cv2.cvtColor(robert.astype(np.uint8), cv2.COLOR_GRAY2BGR)
        self.display_image(self.processed_image, self.canvas_processed)

        
    def laplacian(self):
        self.prepare_for_operation()
    
        # Ubah gambar ke grayscale
        gray = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2GRAY)
    
        # Laplacian untuk deteksi tepi
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    
        # Konversi ke nilai absolut agar negatif menjadi positif
        laplacian = cv2.convertScaleAbs(laplacian)
    
        # Opsional: tingkatkan kontras dengan pengalian faktor (misal 2)
        laplacian = cv2.convertScaleAbs(laplacian * 2)
    
        # Ubah ke BGR agar bisa ditampilkan
        self.processed_image = cv2.cvtColor(laplacian, cv2.COLOR_GRAY2BGR)
        self.display_image(self.processed_image, self.canvas_processed)

    def log(self):
        self.prepare_for_operation()
        
        # Ubah gambar ke grayscale
        gray = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2GRAY)
        
        # Gaussian blur untuk mengurangi noise
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        
        # Laplacian setelah blur
        log = cv2.Laplacian(blurred, cv2.CV_64F)
        
        # Konversi ke nilai absolut
        log = cv2.convertScaleAbs(log)
        
        # Opsional: tingkatkan kontras
        log = cv2.convertScaleAbs(log * 2)
        
        # Ubah ke BGR agar bisa ditampilkan
        self.processed_image = cv2.cvtColor(log, cv2.COLOR_GRAY2BGR)
        self.display_image(self.processed_image, self.canvas_processed)

        
    def canny(self):
        self.prepare_for_operation()
        gray = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2GRAY)
        threshold1 = tk.simpledialog.askinteger("Canny", "Enter threshold1:", 
                                               minvalue=0, maxvalue=255, initialvalue=50)
        threshold2 = tk.simpledialog.askinteger("Canny", "Enter threshold2:", 
                                               minvalue=0, maxvalue=255, initialvalue=150)
        if threshold1 is not None and threshold2 is not None:
            edges = cv2.Canny(gray, threshold1, threshold2)
            self.processed_image = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
            self.display_image(self.processed_image, self.canvas_processed)
    
    def show_compass_menu(self):
        if self.current_image is None:
            messagebox.showwarning("Warning", "Please open an image first!")
            return
            
        menu = tk.Toplevel(self.root)
        menu.title("Compass Edge Detection")
        menu.geometry("300x250")
        
        tk.Label(menu, text="Select Compass Direction:", font=("Arial", 10, "bold")).pack(pady=10)
        
        tk.Button(menu, text="Prewitt Compass (8 directions)", command=lambda: self.compass_edge("prewitt"), width=25).pack(pady=3)
        tk.Button(menu, text="Kirsch Compass (8 directions)", command=lambda: self.compass_edge("kirsch"), width=25).pack(pady=3)
        tk.Button(menu, text="Robinson Compass (8 directions)", command=lambda: self.compass_edge("robinson"), width=25).pack(pady=3)
        
    def compass_edge(self, compass_type):
        self.prepare_for_operation()
        gray = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2GRAY)
        gray = gray.astype(np.float64)
        
        if compass_type == "prewitt":
            kernels = [
            np.array([[-1, 0, 1],
                    [-1, 0, 1],
                    [-1, 0, 1]]),
            np.array([[0, 1, 1],
                    [-1, 0, 1],
                    [-1, -1, 0]]),
            np.array([[1, 1, 1],
                    [0, 0, 0],
                    [-1, -1, -1]]),
            np.array([[1, 1, 0],
                    [1, 0, -1],
                    [0, -1, -1]]),
            np.array([[1, 0, -1],
                    [1, 0, -1],
                    [1, 0, -1]]),
            np.array([[0, -1, -1],
                    [1, 0, -1],
                    [1, 1, 0]]),
            np.array([[-1, -1, -1],
                    [0, 0, 0],
                    [1, 1, 1]]),
            np.array([[-1, -1, 0],
                    [-1, 0, 1],
                    [0, 1, 1]])
            ]

        elif compass_type == "kirsch":
            kernels = [
                np.array([[5, 5, 5], [-3, 0, -3], [-3, -3, -3]]),
                np.array([[-3, 5, 5], [-3, 0, 5], [-3, -3, -3]]),
                np.array([[-3, -3, 5], [-3, 0, 5], [-3, -3, 5]]),
                np.array([[-3, -3, -3], [-3, 0, 5], [-3, 5, 5]]),
                np.array([[-3, -3, -3], [-3, 0, -3], [5, 5, 5]]),
                np.array([[-3, -3, -3], [5, 0, -3], [5, 5, -3]]),
                np.array([[5, -3, -3], [5, 0, -3], [5, -3, -3]]),
                np.array([[5, 5, -3], [5, 0, -3], [-3, -3, -3]])
            ]
        else:
            kernels = [
                np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]),
                np.array([[0, 1, 2], [-1, 0, 1], [-2, -1, 0]]),
                np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]]),
                np.array([[2, 1, 0], [1, 0, -1], [0, -1, -2]]),
                np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]]),
                np.array([[0, -1, -2], [1, 0, -1], [2, 1, 0]]),
                np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]),
                np.array([[-2, -1, 0], [-1, 0, 1], [0, 1, 2]])
            ]
        
        edges = np.zeros_like(gray)
        for kernel in kernels:
            filtered = cv2.filter2D(gray, -1, kernel)
            edges = np.maximum(edges, np.abs(filtered))
        
        edges = cv2.normalize(edges, None, 0, 255, cv2.NORM_MINMAX)
        
        if compass_type == "kirsch":
            edges = cv2.convertScaleAbs(edges, alpha=1.2, beta=10)
        
        self.processed_image = cv2.cvtColor(edges.astype(np.uint8), cv2.COLOR_GRAY2BGR)
        self.display_image(self.processed_image, self.canvas_processed)
            
    # ===== SEGMENTATION =====
    def global_thresholding(self):
        if self.current_image is None:
            messagebox.showwarning("Warning", "Please open an image first!")
            return
        self.prepare_for_operation()
        
        gray = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2GRAY)
        
        threshold_window = tk.Toplevel(self.root)
        threshold_window.title("Global Thresholding")
        threshold_window.geometry("350x300")
        
        tk.Label(threshold_window, text="Select Thresholding Method:", font=("Arial", 10, "bold")).pack(pady=10)
        
        def apply_binary():
            threshold_value = tk.simpledialog.askinteger("Binary Threshold", "Enter threshold value (0-255):", 
                                                        minvalue=0, maxvalue=255, initialvalue=127)
            if threshold_value is not None:
                _, thresh = cv2.threshold(gray, threshold_value, 255, cv2.THRESH_BINARY)
                self.processed_image = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
                self.display_image(self.processed_image, self.canvas_processed)
                threshold_window.destroy()
        
        def apply_binary_inv():
            threshold_value = tk.simpledialog.askinteger("Binary Inverse", "Enter threshold value (0-255):", 
                                                        minvalue=0, maxvalue=255, initialvalue=127)
            if threshold_value is not None:
                _, thresh = cv2.threshold(gray, threshold_value, 255, cv2.THRESH_BINARY_INV)
                self.processed_image = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
                self.display_image(self.processed_image, self.canvas_processed)
                threshold_window.destroy()
        
        def apply_otsu():
            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            self.processed_image = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
            self.display_image(self.processed_image, self.canvas_processed)
            threshold_window.destroy()
        
        def apply_adaptive_mean():
            thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
                                          cv2.THRESH_BINARY, 11, 2)
            self.processed_image = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
            self.display_image(self.processed_image, self.canvas_processed)
            threshold_window.destroy()
        
        def apply_adaptive_gaussian():
            thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                          cv2.THRESH_BINARY, 11, 2)
            self.processed_image = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
            self.display_image(self.processed_image, self.canvas_processed)
            threshold_window.destroy()
        
        tk.Button(threshold_window, text="Binary Threshold", command=apply_binary, width=25).pack(pady=5)
        tk.Button(threshold_window, text="Binary Inverse", command=apply_binary_inv, width=25).pack(pady=5)
        tk.Button(threshold_window, text="Otsu's Threshold", command=apply_otsu, width=25).pack(pady=5)
        tk.Button(threshold_window, text="Adaptive Mean", command=apply_adaptive_mean, width=25).pack(pady=5)
        tk.Button(threshold_window, text="Adaptive Gaussian", command=apply_adaptive_gaussian, width=25).pack(pady=5)
    
    def region_growing(self):
        if self.current_image is None:
            messagebox.showwarning("Warning", "Please open an image first!")
            return
        self.prepare_for_operation()
            
        gray = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        
        threshold = tk.simpledialog.askinteger("Region Growing", 
                                              "Enter threshold value (1-50):", 
                                              minvalue=1, maxvalue=50, initialvalue=15)
        if threshold is None:
            return
        
        seed = (h//2, w//2)
        
        segmented = np.zeros_like(gray)
        visited = np.zeros_like(gray, dtype=bool)
        seed_value = int(gray[seed])
        
        from collections import deque
        queue = deque([seed])
        visited[seed] = True
        
        while queue:
            x, y = queue.popleft()
            
            if abs(int(gray[x, y]) - seed_value) <= threshold:
                segmented[x, y] = 255
                
                for dx, dy in [(1,0), (-1,0), (0,1), (0,-1)]:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < h and 0 <= ny < w and not visited[nx, ny]:
                        visited[nx, ny] = True
                        queue.append((nx, ny))
        
        result = self.current_image.copy()
        result[segmented == 255] = [0, 255, 0]
        
        self.processed_image = result
        self.display_image(self.processed_image, self.canvas_processed)
        
    def watershed(self):
        if self.current_image is None:
            messagebox.showwarning("Warning", "Please open an image first!")
            return
        self.prepare_for_operation()
            
        gray = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2GRAY)
        
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        kernel = np.ones((3,3), np.uint8)
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
        
        sure_bg = cv2.dilate(opening, kernel, iterations=3)
        
        dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
        _, sure_fg = cv2.threshold(dist_transform, 0.5*dist_transform.max(), 255, 0)
        
        sure_fg = np.uint8(sure_fg)
        unknown = cv2.subtract(sure_bg, sure_fg)
        
        _, markers = cv2.connectedComponents(sure_fg)
        markers = markers + 1
        markers[unknown == 255] = 0
        
        markers = cv2.watershed(self.current_image, markers)
        
        result = self.current_image.copy()
        
        result[markers == -1] = [0, 0, 255]
        
        unique_markers = np.unique(markers)
        colors = np.random.randint(0, 255, size=(len(unique_markers), 3))
        
        for i, marker in enumerate(unique_markers):
            if marker > 1:
                result[markers == marker] = colors[i]
        
        self.processed_image = result
        self.display_image(self.processed_image, self.canvas_processed)
        
    def edge_based_segmentation(self):
        if self.current_image is None:
            messagebox.showwarning("Warning", "Please open an image first!")
            return
        self.prepare_for_operation()
        
        edge_seg_window = tk.Toplevel(self.root)
        edge_seg_window.title("Edge Based Segmentation")
        edge_seg_window.geometry("350x250")
        
        tk.Label(edge_seg_window, text="Select Edge Detection Method:", font=("Arial", 10, "bold")).pack(pady=10)
        
        def apply_canny_seg():
            gray = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2GRAY)
            
            edges = cv2.Canny(gray, 50, 150)
            
            kernel = np.ones((3,3), np.uint8)
            edges_dilated = cv2.dilate(edges, kernel, iterations=2)
            
            result = self.current_image.copy()
            result[edges_dilated > 0] = [0, 255, 255]
            
            self.processed_image = result
            self.display_image(self.processed_image, self.canvas_processed)
            edge_seg_window.destroy()
        
        def apply_sobel_seg():
            gray = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2GRAY)
            
            sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            sobel = np.sqrt(sobelx**2 + sobely**2)
            sobel = cv2.normalize(sobel, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            
            _, edges = cv2.threshold(sobel, 50, 255, cv2.THRESH_BINARY)
            
            result = self.current_image.copy()
            result[edges > 0] = [255, 0, 255]
            
            self.processed_image = result
            self.display_image(self.processed_image, self.canvas_processed)
            edge_seg_window.destroy()
        
        def apply_laplacian_seg():
            gray = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2GRAY)
            
            blurred = cv2.GaussianBlur(gray, (3, 3), 0)
            
            laplacian = cv2.Laplacian(blurred, cv2.CV_64F)
            laplacian = np.absolute(laplacian)
            laplacian = cv2.normalize(laplacian, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            
            _, edges = cv2.threshold(laplacian, 30, 255, cv2.THRESH_BINARY)
            
            result = self.current_image.copy()
            result[edges > 0] = [255, 255, 0]
            
            self.processed_image = result
            self.display_image(self.processed_image, self.canvas_processed)
            edge_seg_window.destroy()
        
        def apply_morphological_seg():
            gray = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2GRAY)
            
            edges = cv2.Canny(gray, 50, 150)
            
            kernel = np.ones((5,5), np.uint8)
            closed_edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=3)
            
            contours, _ = cv2.findContours(closed_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            result = np.zeros_like(self.current_image)
            
            for i, contour in enumerate(contours):
                if cv2.contourArea(contour) < 500:
                    continue
                
                color = np.random.randint(50, 255, size=3).tolist()
                cv2.drawContours(result, [contour], -1, color, -1)
                cv2.drawContours(result, [contour], -1, (255, 255, 255), 2)
            
            self.processed_image = result
            self.display_image(self.processed_image, self.canvas_processed)
            edge_seg_window.destroy()
        
        tk.Button(edge_seg_window, text="Canny Edge Segmentation", command=apply_canny_seg, width=30).pack(pady=5)
        tk.Button(edge_seg_window, text="Sobel Edge Segmentation", command=apply_sobel_seg, width=30).pack(pady=5)
        tk.Button(edge_seg_window, text="Laplacian Edge Segmentation", command=apply_laplacian_seg, width=30).pack(pady=5)
        tk.Button(edge_seg_window, text="Morphological Segmentation", command=apply_morphological_seg, width=30).pack(pady=5)
            
    # ===== ABOUT =====
    def show_info(self):
        info_window = tk.Toplevel(self.root)
        info_window.title("Info Tim Developer")
        info_window.geometry("400x300")
        
        info_text = """
        Aplikasi Pengolahan Citra Digital
        
        Dikembangkan oleh:
        IMKTI
        
        Anggota Tim:
        1. Christian Elbert
        2. Davit Martua Marpaung
        3. Ignasius David Christian Hasugian
        
        Versi: 1.0
        Tahun: 2025
        
        Aplikasi ini dibuat untuk memenuhi tugas
        mata kuliah Pengolahan Citra Digital
        """
        
        tk.Label(info_window, text=info_text, justify=tk.LEFT, 
                font=("Arial", 10)).pack(padx=20, pady=20)
        
    def show_tutorial(self):
        tutorial_window = tk.Toplevel(self.root)
        tutorial_window.title("Tutorial")
        tutorial_window.geometry("500x400")
        
        tutorial_text = """
        TUTORIAL PENGGUNAAN APLIKASI
        
        1. Membuka Gambar:
           - Klik File > Open
           - Pilih file gambar (JPG, PNG, BMP, TIFF)
        
        2. Memilih Mode Operasi:
           ✓ NORMAL MODE (Default):
             - Hasil operasi muncul di panel kanan
             - Klik "Apply to Original" untuk stack efek
             - Cocok untuk mencoba-coba fitur
           
           ✓ STACK MODE (Centang checkbox):
             - Hasil operasi muncul di panel kanan DULU
             - Ketika operasi BERIKUTNYA dilakukan, hasil sebelumnya
               otomatis jadi input untuk operasi baru
             - Tidak perlu klik "Apply to Original"
             - Cocok untuk workflow cepat
             - Contoh: Crop (lihat hasil) → Brightness (auto-apply + proses) → 
                       Sobel (auto-apply + proses)
        
        3. Memproses Gambar:
           - Pilih menu operasi yang diinginkan
           - Gambar hasil akan muncul di panel kanan
        
        4. Stacking Effects (2 Cara):
           Cara 1 - NORMAL MODE:
           - Aplikasikan efek pertama (misal: Cropping)
           - Klik tombol "Apply to Original ➜"
           - Aplikasikan efek kedua (misal: Edge Detection)
           - Ulangi untuk efek tambahan
           
           Cara 2 - STACK MODE (RECOMMENDED):
           - Centang "Stack Mode (Auto-Apply)"
           - Aplikasikan efek berturut-turut tanpa klik Apply
           - Setiap operasi langsung jadi input untuk operasi berikutnya
        
        5. Reset Gambar:
           - Klik tombol "⟲ Reset" untuk kembali ke gambar asli
           - Berguna jika ingin mulai dari awal
        
        6. Menyimpan Hasil:
           - Klik File > Save atau Save As
           - Pilih lokasi dan format file
        
        Link Tutorial & Dokumentasi:
        ✓ GitHub: https://github.com/imkti/image-processing
        ✓ Video Tutorial: https://youtube.com/imkti-tutorial
        
        Tips:
        - Gunakan STACK MODE untuk workflow cepat
        - Gunakan NORMAL MODE untuk eksplorasi fitur
        - Gunakan "Reset" untuk kembali ke gambar asli
        - Beberapa operasi memerlukan input parameter
        - Gunakan Save As untuk menyimpan dengan nama berbeda
        
        FITUR BARU:
        ✓ 2 Mode Operasi (Normal & Stack)
        ✓ Stack Mode dengan Auto-Apply
        ✓ Stacking/Chaining Effects - gabungkan beberapa filter
        ✓ Reset Button - kembali ke gambar asli
        ✓ Convolution dengan berbagai kernel
        ✓ Cropping manual dengan persentase
        ✓ Compass Edge Detection (Prewitt, Kirsch, Robinson)
        ✓ Edge Based Segmentation dengan 4 metode
        ✓ Zooming yang sudah diperbaiki
        """
        
        text_widget = tk.Text(tutorial_window, wrap=tk.WORD, font=("Arial", 10))
        text_widget.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        text_widget.insert(tk.END, tutorial_text)
        text_widget.config(state=tk.DISABLED)

# Import tambahan untuk dialog
import tkinter.simpledialog

if __name__ == "__main__":
    root = tk.Tk()
    app = ImageProcessingApp(root)
    root.mainloop()