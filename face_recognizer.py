from face_functions_2 import *
#
import argparse
#from tkinter import Tk, filedialog

'''
def select_image_file():
    """Open a file dialog to select an image file"""
    root = Tk()
    root.withdraw()  # Hide the main window
    file_path = filedialog.askopenfilename(
        title="Select an image file",
        filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")]
    )
    root.destroy()
    return file_path
'''


def main_cli():
    """Command line interface for the program"""
    parser = argparse.ArgumentParser(description="Face restoration and recognition tool")
    parser.add_argument("--image", help="Path to the image file (optional, will open file dialog if not provided)")
    parser.add_argument("--known", default="known_people", help="Directory with known people images")
    parser.add_argument("--tolerance", type=float, default=0.5, help="Face recognition tolerance (lower is stricter)")
    parser.add_argument("--cpus", type=int, default=-1, help="Number of CPU cores to use (-1 for all)")
    parser.add_argument("--show-distance", default=False, help="Show distance information in results")
    
    args = parser.parse_args()
    
    # Get the image path from command line or file dialog
    '''image_path = args.image
    if not image_path:
        print("Please select an image file...")
        #image_path = 'GFPGAN/inputs/upload/'
        
    if not image_path:
        #print("No image selected. Exiting.")
        return'
    '''
    # Process the selected image
    process_image(
        input_image_path=args.image, 
        known_people_dir=args.known,
        tolerance=args.tolerance,
        cpus=args.cpus,
        show_distance=args.show_distance
    )


if __name__ == "__main__":
    main_cli()
