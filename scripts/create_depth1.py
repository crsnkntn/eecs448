from util.dataset import filter_directory

input_dir = 'arctic-shift-dataset/posts'
output_dir = 'datasets/1D_20C_10W/by_post'

filter_directory(input_dir, output_dir, n_comments=20, n_words=10)
