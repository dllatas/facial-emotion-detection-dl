import os
import cv2

# First check how many faces are detected per frame; report when value is different than one

def save_image(image, dest_path, filename):
	cv2.imwrite(os.path.join(dest_path, filename), image)

def check_number_faces_detected(face, faces_to_detect, filename):
	if len(face) != faces_to_detect:
		print "Faces found: " + str(len(face)) + " on file " + filename
		return True
	return False

def show_image(image, header="faces"):
	cv2.imshow(header, image)
	cv2.waitKey(0)

def crop_image(face, image):
	for (x, y, w, h) in face:
		crop_image = image[y:y+h, x:x+w]
	return crop_image

def draw_rectangle(face, image):
	for (x, y, w, h) in face:
		cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
	return image

def get_face(filename, face_cascade):
	image = cv2.imread(filename)
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	face = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=2, minSize=(70, 70))
	return face, image

def get_images(path, face_cascade, dest_path, faces_to_detect):
	erros = 0
	for root, dirs, files in os.walk(path, True):
		for name in files:
			face = get_face(os.path.join(root, name), face_cascade)
			check = check_number_faces_detected(face[0], faces_to_detect, os.path.join(root, name))
			if check is True:
				erros = erros + 1
			else:
				image = crop_image(face[0], face[1])
				save_image(image, dest_path, name)
	print erros
			
def main(argv=None):  # pylint: disable=unused-argument
	face_cascade = cv2.CascadeClassifier("/home/neo/opencv-3.1.0/data/haarcascades/haarcascade_frontalface_default.xml")
	#image_path = "/home/neo/projects/deepLearning/data/image_exp2/"
	#image_path = "/home/neo/projects/deepLearning/data/ck_image_seq_10"
	image_path = "/home/neo/projects/deepLearning/data/amfed/happy"
	#dest_path = "/home/neo/projects/deepLearning/data/crop_faces_seq_10/"
	dest_path = "/home/neo/projects/deepLearning/data/amfed_faces"
	faces_to_detect = 1
	get_images(image_path, face_cascade, dest_path, faces_to_detect)

if __name__ == '__main__':
	main()