from basket_inference import infer_one_basket
import time

def count_fruits(file_path):
    fresh_fruits = 0
    rotten_fruits = 0

    with open(file_path, 'r') as file:
        for line in file:
            if line.startswith('empty_baskets/'):
                continue
            parts = line.strip().split(':')
            if len(parts) == 2:
                fruit_type, count = parts
                if fruit_type in ['freshapples', 'freshoranges']:
                    fresh_fruits += int(count)
                elif fruit_type in ['rottenapples', 'rottenoranges']:
                    rotten_fruits += int(count)

    return fresh_fruits, rotten_fruits

def main():
    test_baskets_dir = 'baskets/test_baskets'

    checkpoints = {'vit_b': 'sam_vit_b_01ec64.pth', 'vit_l': 'sam_vit_l_0b3195.pth', 'vit_h': 'sam_vit_h_4b8939.pth'}
    avg_time = {'vit_b': 0, 'vit_l': 0, 'vit_h': 0}
    pmae = {'vit_b': 0, 'vit_l': 0, 'vit_h': 0}

    for model in ['vit_b', 'vit_l', 'vit_h']:
        start_time = time.time()
        pmae_sum = 0
        successful_counts = 0
        for i in range(100):
            image_path = test_baskets_dir + f"/image/{i}.png"
            fpos_path = test_baskets_dir + f"/fpos/{i}.np.npy"
            metadata_path = test_baskets_dir + f"/meta/{i}.txt"
            
            actual_fresh, actual_rotten = count_fruits(metadata_path)
            
            try:
                inferred_fresh, inferred_rotten = infer_one_basket(test_baskets_dir, 
                                                                    i, sam_model=model, 
                                                                    checkpoint_path=checkpoints[model])
                successful_counts += 1
                pmae_sum += abs(actual_rotten - inferred_rotten) / (actual_fresh + actual_rotten)
            except:
                pass
        
        end_time = time.time()
        time_sum = end_time - start_time
        
        avg_time[model] = time_sum / successful_counts
        pmae[model] = pmae_sum / successful_counts

        print(f"Model: {model}")
        print(f"Average time: {avg_time[model]}")
        print(f"PMAE: {pmae[model] * 100:.2f}%")
        print(f"Successful counts: {successful_counts}\n")

        filename = "results.txt"

        with open(filename, "a") as file:
            file.write(f"Model: {model}\n")
            file.write(f"Average time: {avg_time[model]}\n")
            file.write(f"PMAE: {pmae[model] * 100:.2f}%\n")
            file.write(f"Successful counts: {successful_counts}\n\n")

main()