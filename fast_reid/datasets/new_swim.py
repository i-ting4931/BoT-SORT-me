import os
import argparse
import cv2
import numpy as np

def generate_trajectories(file_path, groundTrues):
    try:
        with open(file_path, 'r') as f:
            lines = f.read().split('\n')
            values = []
            for l in lines:
                split = l.split(',')
                if len(split) < 2:
                    break
                numbers = [float(i) for i in split]
                values.append(numbers)

            values = np.array(values, np.float_)

            if groundTrues:
                # Uncomment and modify as needed for filtering criteria
                # values = values[values[:, 6] == 1, :]  # Remove ignore objects
                # values = values[values[:, 7] == 1, :]  # Specific object class only
                values = values[values[:, 8] > 0.4, :]  # Visibility threshold

            values[:, 4] += values[:, 2]
            values[:, 5] += values[:, 3]

            return values

    except Exception as e:
        print(f"Error reading trajectories from {file_path}: {e}")
        return None

def make_parser():
    parser = argparse.ArgumentParser("SWIM Dataset ReID Preparation")
    parser.add_argument("--data_path", default="C:\\BoT-SORT\\datasets\\swim", help="Path to SWIM data")
    parser.add_argument("--save_path", default="C:\\BoT-SORT\\fast_reid\\datasets", help="Path to save the SWIM-ReID dataset")
    return parser

def process_sequence(data_path, save_path, seq, id_offset, train_save_path, test_save_path):
    print(f"Processing sequence {seq}...")
    print(f"Current id_offset: {id_offset}")

    ground_truth_path = os.path.join(data_path, seq, 'gt', 'gt.txt')
    gt = generate_trajectories(ground_truth_path, groundTrues=True)

    if gt is None:
        return id_offset

    images_path = os.path.join(data_path, seq, 'img1')
    img_files = os.listdir(images_path)
    img_files.sort()

    num_frames = len(img_files)
    max_id_per_seq = 0

    for f in range(num_frames):
        img_path = os.path.join(images_path, img_files[f])
        print(f"Processing frame {f}: {img_path}")

        img = cv2.imread(img_path)

        if img is None:
            print(f"ERROR: Failed to read image {img_path}")
            continue

        H, W, _ = img.shape

        det = gt[f + 1 == gt[:, 0], 1:].astype(np.int_)

        for d in range(np.size(det, 0)):
            id_ = det[d, 0]
            x1 = det[d, 1]
            y1 = det[d, 2]
            x2 = det[d, 3]
            y2 = det[d, 4]

            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(x2, W)
            y2 = min(y2, H)

            patch = img[y1:y2, x1:x2, :]

            max_id_per_seq = max(max_id_per_seq, id_)

            fileName = f"{str(id_ + id_offset).zfill(7)}_{seq}_{str(f).zfill(7)}_acc_data.bmp"

            if f < num_frames // 2:
                save_file = os.path.join(train_save_path, fileName)
            else:
                save_file = os.path.join(test_save_path, fileName)

            cv2.imwrite(save_file, patch)
            print(f"Saved patch: {save_file}")

    return id_offset + max_id_per_seq

def main(args):
    try:
        # Create folder for outputs
        save_path = os.path.join(args.save_path, 'SWIM-ReID')
        os.makedirs(save_path, exist_ok=True)

        train_save_path = os.path.join(save_path, 'bounding_box_train')
        os.makedirs(train_save_path, exist_ok=True)
        
        test_save_path = os.path.join(save_path, 'bounding_box_test')
        os.makedirs(test_save_path, exist_ok=True)

        # Process training data
        train_data_path = os.path.join(args.data_path, 'train')
        train_seqs = os.listdir(train_data_path)
        train_seqs.sort()

        id_offset = 0

        for seq in train_seqs:
            id_offset = process_sequence(train_data_path, save_path, seq, id_offset, train_save_path, test_save_path)

        print("Dataset generation completed.")

    except Exception as e:
        print(f"Error during dataset generation: {e}")

if __name__ == "__main__":
    args = make_parser().parse_args()
    main(args)
