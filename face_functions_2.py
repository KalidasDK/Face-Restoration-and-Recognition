from __future__ import print_function
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import os
import re
import click
import PIL.Image
import face_recognition.api as face_recognition
import multiprocessing
import subprocess
import itertools
import sys

def image_files_in_folder(folder):
    return [os.path.join(folder, f) for f in os.listdir(folder) if re.match(r'.*\.(jpg|jpeg|png)', f, flags=re.I)]


def print_result(name):
    print("Person identified: {}".format(name))



def scan_known_people(known_people_folder):
    known_names = []
    known_face_encodings = []

    for file in image_files_in_folder(known_people_folder):
        basename = os.path.splitext(os.path.basename(file))[0]
        img = face_recognition.load_image_file(file)
        encodings = face_recognition.face_encodings(img)

        if len(encodings) > 1:
            click.echo("WARNING: More than one face found in {}. Only considering the first face.".format(file))

        if len(encodings) == 0:
            click.echo("WARNING: No faces found in {}. Ignoring file.".format(file))
        else:
            known_names.append(basename)
            known_face_encodings.append(encodings[0])

    return known_names, known_face_encodings




def test_image(input_image_path, known_face_names, known_face_encodings, tolerance=0.51, show_distance=False):
    unknown_image = face_recognition.load_image_file(input_image_path)
    input_filename = os.path.basename(input_image_path)

    face_locations = face_recognition.face_locations(unknown_image)
    face_encodings = face_recognition.face_encodings(unknown_image, face_locations)

    # Convert the image to a PIL-format image so that we can draw on top of it with the Pillow library
    # See http://pillow.readthedocs.io/ for more about PIL/Pillow
    pil_image = Image.fromarray(unknown_image)
    # Create a Pillow ImageDraw Draw instance to draw with
    draw = ImageDraw.Draw(pil_image)

    # Loop through each face found in the unknown image
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        # See if the face is a match for the known face(s)
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)

        name = "Unknown"

        # Or instead, use the known face with the smallest distance to the new face
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = known_face_names[best_match_index]
            print_result(name)

        # Draw a box around the face using the Pillow module
        draw.rectangle(((left, top), (right, bottom)), outline=(0, 0, 255), width=5)

        # Draw a label with a name below the face
        #text_bbox = draw.textbbox((0, 0), name)
        text_height = 30
        font = ImageFont.load_default(size = 40)
        text_width, text_height = draw.textbbox((0, 0), name, font=font)[2:]
        padding = 20
        box_width = text_width + padding
        #text_width, text_height = draw.textsize(name, font=font)
        
        draw.rectangle(((left, bottom - text_height - 10), (left + box_width, bottom)), fill=(0, 0, 255), outline=(0, 0, 255))
        draw.text((left+ padding/2, bottom - text_height - 5), name, fill=(255, 255, 255, 255), font=font)

        # Remove the drawing library from memory as per the Pillow docs
    del draw

    # Display the resulting image
    #pil_image.show()
    # You can also save a copy of the new image to disk if you want by uncommenting this line
    pil_image.save(f"GFPGAN/results/restored_imgs/{input_filename}")



def process_images_in_process_pool(images_to_check, known_names, known_face_encodings, number_of_cpus, tolerance, show_distance):
    if number_of_cpus == -1:
        processes = None
    else:
        processes = number_of_cpus

    # macOS will crash due to a bug in libdispatch if you don't use 'forkserver'
    context = multiprocessing
    if "forkserver" in multiprocessing.get_all_start_methods():
        context = multiprocessing.get_context("forkserver")

    pool = context.Pool(processes=processes)
    function_parameters = zip(
        images_to_check,
        itertools.repeat(known_names),
        itertools.repeat(known_face_encodings),
        itertools.repeat(tolerance),
        itertools.repeat(show_distance)
    )

    pool.starmap(test_image, function_parameters)


def main(known_people_folder, image_to_check, cpus, tolerance, show_distance):
    known_names, known_face_encodings = scan_known_people(known_people_folder)

    # Multi-core processing only supported on Python 3.4 or greater
    if (sys.version_info < (3, 4)) and cpus != 1:
        click.echo("WARNING: Multi-processing support requires Python 3.4 or greater. Falling back to single-threaded processing!")
        cpus = 1

    if os.path.isdir(image_to_check):
        if cpus == 1:
            [test_image(image_file, known_names, known_face_encodings, tolerance, show_distance) for image_file in image_files_in_folder(image_to_check)]
        else:
            process_images_in_process_pool(image_files_in_folder(image_to_check), known_names, known_face_encodings, cpus, tolerance, show_distance)
    else:
        test_image(image_to_check, known_names, known_face_encodings, tolerance, show_distance)



def process_image(input_image_path, known_people_dir, tolerance, cpus, show_distance):

    """Process an image through restoration and face recognition"""
    # Create necessary directories if they don't exist
    os.makedirs("GFPGAN/inputs/upload", exist_ok=True)
    os.makedirs("GFPGAN/results", exist_ok=True)

    # Copy the input image to the GFPGAN input directory
    input_filename = os.path.basename(input_image_path)
    
    # Construct the output path
    print("Input filename= ", input_filename)
    output_filename = input_filename
    restored_image_path = f"GFPGAN/results/restored_imgs/{output_filename}"

    
    # Run the restoration process
    restorer = f"python3 GFPGAN/inference_gfpgan.py -i GFPGAN/inputs/upload -o GFPGAN/results -v 1.3 -s 2 --bg_upsampler realesrgan"

    try:
        print("Starting image restoration...")
        subprocess.run(restorer, shell=True, check=True)
        print(f"Image restored successfully: {restored_image_path}")
    except subprocess.CalledProcessError as e:
        print(f"Error in restoration: {e}")
        return
    
    # Run face recognition on the restored image
    print("Starting face recognition...")
    main(known_people_dir, restored_image_path, cpus=cpus, tolerance=tolerance, show_distance=show_distance)
    print("Processing complete!")
