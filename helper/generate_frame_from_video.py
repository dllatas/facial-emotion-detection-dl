import csv
import cv2
import os

"""
1 (e) Angry    - AU 4+5+15+17, 	 (4/5)
2 (f) Contempt - AU 14, 		 (1/1)
3 (a) Disgust  - AU 1+4+15+17, 	 (3/4)
4 (d) Fear     - AU 1+4+7+20, 	 (1/4)
5 (b) Happy    - AU 6+12+25, 	 (Smile)
6 (g) Sadness  - AU 1+2+4+15+17  (4/5)
7 (c) Surprise - AU 1+2+5+25+27  (2/5)
SURPRISE AND FEAR GG
"""

def change_to_video_name(csv_name, suffix):
	return csv_name[:-10]+"."+suffix

def generate_frame(video_path, video_name, second, label, dest_path):
	vidcap = cv2.VideoCapture(os.path.join(video_path, video_name))
	vidcap.set(0, int(second*1000))
	success, image = vidcap.read()
	if success:
		cv2.imwrite(os.path.join(dest_path, video_name+"_"+str(second)+"_"+str(label)+".jpg"), image)

def check_angry(content):
	baseline = 50
	#disgust = ["AU4", "AU15", "AU17"]
	sadness = ["AU2", "AU4", "AU15", "AU17"]
	#angry = ["AU4", "AU5", "AU15", "AU17"]
	label = 1
	emotion_time = content[0][1]
	emotion = []
	for c in content:
		for h in sadness:
			if c[0] == h:
				emotion.append(c[1])
	print emotion
	factor = sum(emotion)/len(sadness)
	if factor >= baseline:
		return emotion_time, label

def check_contempt(content):
	baseline = 100
	contempt = ["AU14"]
	label = 2
	emotion_time = content[0][1]
	for c in content:
		for h in contempt:
			if c[0] == h and c[1] >= baseline:
				return emotion_time, label

def check_happiness(content):
	baseline = 100
	happiness = ["Smile"]
	label = 5
	emotion_time = content[0][1]
	for c in content:
		for h in happiness:
			if c[0] == h and c[1] >= baseline:
				return emotion_time, label

def get_content(header, row):
	content = row[0].split(',')
	result = []
	for h in header:
		result.append([h[0], float(content[h[1]])])
	return result

def get_header_au(row):
	rules = ["Time", "Smile", "AU"]
	header = row[0].split(',')
	result = []
	i = 0
	for h in header:
		if h in rules or h[0:2] in rules:
			result.append([h, i])
		i = i + 1
	return result

def process_video(csv_path, video_path, dest_path, suffix):
	for root, dirs, files in os.walk(csv_path, True):
		for name in files:
			with open(os.path.join(root, name), 'rU') as csvfile:
				reader = csv.reader(csvfile, delimiter=' ', quotechar='|')
				for row in reader:
					if reader.line_num == 1:
						header = get_header_au(row)
					else:
						if len(header) > 1:
							content = get_content(header, row)
							emotion = check_angry(content)
							if emotion is not None:
								generate_frame(video_path, change_to_video_name(name, suffix), emotion[0], emotion[1], dest_path)

def main(argv=None):  # pylint: disable=unused-argument
	csv_path = "/home/neo/projects/deepLearning/data/au_labels"
	video_path = "/home/neo/projects/deepLearning/data/videos"
	dest_path = "/home/neo/projects/deepLearning/data/amfed/sadness"
	suffix = "flv"
	process_video(csv_path, video_path, dest_path, suffix)
	

if __name__ == '__main__':
	main()