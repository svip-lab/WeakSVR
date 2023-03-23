## Pre-requisities: run 'pip install youtube-dl' to install the youtube-dl package.
## Specify your location of output videos and input json file.
import json
import os

output_path = 'videos'
json_path = 'COIN.json'
ids_path = 'all_ids.txt'

if not os.path.exists(output_path):
	os.mkdir(output_path)

data = json.load(open(json_path, 'r'))['database']
# youtube_ids = list(Data.keys())
youtube_ids = []

with open(ids_path, 'r') as f:
	for line in f.readlines():
		line = line.strip('\n')
		youtube_ids.append(line)

count = 0
for youtube_id in youtube_ids:


	count += 1
	print('***** [ %d / %d ] *****' % (count, len(youtube_ids)))

	info = data[youtube_id]
	type = info['recipe_type']
	url = info['video_url']
	vid_loc = output_path + '/' + str(type)
	if not os.path.exists(vid_loc):
		os.mkdir(vid_loc)
	if not os.path.exists(os.path.join(vid_loc, youtube_id + '.mp4')):
		print('downloading %s' % os.path.join(vid_loc, youtube_id + '.mp4'))
		os.system('youtube-dl -o ' + vid_loc + '/' + youtube_id + '.mp4' + ' -f 134 ' + url)
	else:
		print(os.path.join(vid_loc, youtube_id + '.mp4') + ' already exists!')


# To save disk space, you could download the best format available
# 	but not better that 480p or any other qualities optinally
# See https://askubuntu.com/questions/486297/how-to-select-video-quality-from-youtube-dl
