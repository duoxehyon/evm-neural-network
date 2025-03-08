import torch
from torchvision import datasets, transforms
import numpy as np
from web3 import Web3
import json
import time
import argparse
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import matplotlib.pyplot as plt

class MNISTOnChainTester:
    def __init__(self, contract_address, rpc_url, account_address=None, get_eth_price=True, skip_gas=False):
        self.w3 = Web3(Web3.HTTPProvider(rpc_url))
        if not self.w3.is_connected():
            raise Exception(f"Failed to connect to blockchain via {rpc_url}")
        
        print(f"Connected to network: Chain ID {self.w3.eth.chain_id}")
        
        self.skip_gas = skip_gas
        if skip_gas:
            print("Gas estimation disabled")
        
        self.contract_abi = [
            {"inputs":[{"internalType":"uint8[400]","name":"inputImage","type":"uint8[400]"}],"name":"predict","outputs":[{"internalType":"uint8","name":"","type":"uint8"}],"stateMutability":"nonpayable","type":"function"},
            {"inputs":[],"name":"isModelReady","outputs":[{"internalType":"bool","name":"","type":"bool"}],"stateMutability":"view","type":"function"}
        ]
        
        self.contract_address = Web3.to_checksum_address(contract_address)
        self.contract = self.w3.eth.contract(address=self.contract_address, abi=self.contract_abi)
        
        if account_address:
            self.account_address = Web3.to_checksum_address(account_address)
        else:
            self.account_address = Web3.to_checksum_address("0x0000000000000000000000000000000000000000")
        
        self.gas_price = self.w3.eth.gas_price
        
        self.eth_price = None
        if get_eth_price and not skip_gas:
            try:
                response = requests.get("https://api.coingecko.com/api/v3/simple/price?ids=ethereum&vs_currencies=usd")
                self.eth_price = response.json()["ethereum"]["usd"]
                print(f"Current ETH price: ${self.eth_price}")
            except Exception as e:
                print(f"Warning: Could not fetch ETH price: {e}")
        
        try:
            is_ready = self.contract.functions.isModelReady().call()
            if not is_ready:
                raise Exception("Model is not fully deployed and ready for inference")
            print(f"Model is ready for inference at {self.contract_address}")
        except Exception as e:
            raise Exception(f"Error connecting to contract: {e}")
        
        self.test_dataset = self.load_test_dataset()
        print(f"Loaded {len(self.test_dataset)} test images")
    
    def load_test_dataset(self):
        transform = transforms.Compose([
            transforms.Resize((20, 20)),
            transforms.ToTensor()
        ])
        
        try:
            test_dataset = datasets.MNIST('data', train=False, download=True, transform=transform)
            return test_dataset
        except Exception as e:
            raise Exception(f"Error loading MNIST test dataset: {e}")
    
    def preprocess_image(self, image_tensor):
        image_np = (image_tensor.squeeze().numpy() * 255).astype(np.uint8)
        flattened = image_np.flatten().tolist()
        return flattened
    
    def predict_onchain(self, image_idx):
        try:
            image, label = self.test_dataset[image_idx]
            preprocessed_image = self.preprocess_image(image)
            prediction = self.contract.functions.predict(preprocessed_image).call()
            
            result = {
                'index': image_idx,
                'prediction': prediction,
                'label': label,
                'correct': prediction == label,
                'image': image,
            }
            
            if not self.skip_gas:
                try:
                    estimated_gas = self.contract.functions.predict(preprocessed_image).estimate_gas({
                        'from': self.account_address,
                        'gas': 50000000
                    })
                    
                    cost_in_eth = self.w3.from_wei(estimated_gas * self.gas_price, 'ether')
                    cost_in_usd = cost_in_eth * self.eth_price if self.eth_price else None
                    
                    result.update({
                        'gas_used': estimated_gas,
                        'cost_eth': cost_in_eth,
                        'cost_usd': cost_in_usd
                    })
                except Exception as e:
                    print(f"Gas estimation failed for image {image_idx}: {e}")
                    
                    fixed_gas = 5000000
                    result.update({
                        'gas_used': fixed_gas,
                        'gas_estimated': False,
                        'gas_error': str(e)
                    })
                    
                    if self.eth_price:
                        cost_in_eth = self.w3.from_wei(fixed_gas * self.gas_price, 'ether')
                        cost_in_usd = cost_in_eth * self.eth_price
                        result.update({
                            'cost_eth': cost_in_eth,
                            'cost_usd': cost_in_usd
                        })
            
            return result
            
        except Exception as e:
            print(f"Error predicting image {image_idx}: {e}")
            return {
                'index': image_idx,
                'error': str(e)
            }
    
        def visualize_results(self, results):
        if not results:
            print("No results to visualize")
            return
        
        digit_examples = {}
        
        for r in results:
            if 'error' not in r and r.get('correct', False):
                digit = r['label']  # Use true label
                if digit not in digit_examples and 0 <= digit <= 9:
                    digit_examples[digit] = r
        
        for r in results:
            if 'error' not in r and not r.get('correct', False):
                digit = r['label']  # Use true label
                if digit not in digit_examples and 0 <= digit <= 9:
                    digit_examples[digit] = r
        
        fig, axes = plt.subplots(2, 5, figsize=(15, 6))
        axes = axes.flatten()
        
        for digit in range(10):
            ax = axes[digit]
            
            if digit in digit_examples:
                result = digit_examples[digit]
                ax.imshow(result['image'].squeeze(), cmap='gray')
                
                if result.get('correct', False):
                    title = f"Correct: {result['prediction']}"
                    color = 'green'
                else:
                    title = f"Pred: {result['prediction']}, True: {digit}"
                    color = 'red'
                
                if 'gas_used' in result:
                    est_tag = "" if result.get('gas_estimated', True) else "~"
                    title += f"\nGas: {est_tag}{result['gas_used']:,}"
                
                ax.set_title(title, color=color)
            else:
                ax.text(0.5, 0.5, f"No example\nfor digit {digit}", 
                    ha='center', va='center')
            
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xlabel(f"Digit {digit}", fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('mnist_onchain_results.png', dpi=150, bbox_inches='tight')
        print(f"Saved visualization with one example per digit (0-9) to mnist_onchain_results.png")
        
        try:
            plt.show()
        except:
            pass
    
    def visualize_gas_usage(self, results):
        gas_values = [r['gas_used'] for r in results if 'gas_used' in r]
        if not gas_values:
            print("No gas data available for visualization")
            return
            
        plt.figure(figsize=(10, 6))
        plt.hist(gas_values, bins=20, alpha=0.7, color='blue')
        plt.axvline(np.mean(gas_values), color='red', linestyle='dashed', linewidth=2)
        
        estimated_count = sum(1 for r in results if r.get('gas_estimated', True) and 'gas_used' in r)
        approximated_count = len(gas_values) - estimated_count
        
        title = f'Gas Usage Distribution (Mean: {np.mean(gas_values):,.0f})'
        if approximated_count > 0:
            title += f'\n({approximated_count} approximated values, {estimated_count} estimated values)'
            
        plt.title(title)
        plt.xlabel('Gas Used')
        plt.ylabel('Frequency')
        plt.grid(alpha=0.3)
        plt.savefig('gas_usage_histogram.png')
        print("Saved gas usage visualization to 'gas_usage_histogram.png'")
        
        try:
            plt.show()
        except:
            pass
    
    def evaluate_model(self, num_samples=None, max_workers=10):
        if num_samples is None:
            num_samples = len(self.test_dataset)
        else:
            num_samples = min(num_samples, len(self.test_dataset))
        
        print(f"Testing on-chain model with {num_samples} images...")
        
        results = []
        start_time = time.time()
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_idx = {executor.submit(self.predict_onchain, i): i for i in range(num_samples)}
            
            for future in tqdm(as_completed(future_to_idx), total=num_samples, desc="Predictions"):
                idx = future_to_idx[future]
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    print(f"Error processing image {idx}: {e}")
        
        elapsed_time = time.time() - start_time
        
        valid_results = [r for r in results if 'error' not in r]
        error_count = len(results) - len(valid_results)
        
        if valid_results:
            correct_count = sum(1 for r in valid_results if r['correct'])
            accuracy = correct_count / len(valid_results) * 100
            
            print(f"\nResults:")
            print(f"  Total images tested: {len(valid_results)}")
            print(f"  Correct predictions: {correct_count}")
            print(f"  On-chain accuracy: {accuracy:.2f}%")
            print(f"  Errors during testing: {error_count}")
            print(f"  Time taken: {elapsed_time:.2f} seconds")
            print(f"  Average time per image: {elapsed_time/num_samples:.4f} seconds")
            
            gas_values = [r['gas_used'] for r in valid_results if 'gas_used' in r]
            if gas_values and not self.skip_gas:
                estimated_count = sum(1 for r in valid_results if r.get('gas_estimated', True) and 'gas_used' in r)
                approximated_count = len(gas_values) - estimated_count
                
                avg_gas = np.mean(gas_values)
                min_gas = np.min(gas_values)
                max_gas = np.max(gas_values)
                
                print(f"\nGas Usage (based on {estimated_count} estimated, {approximated_count} approximated):")
                print(f"  Average gas per inference: {avg_gas:,.0f}")
                print(f"  Min gas: {min_gas:,}")
                print(f"  Max gas: {max_gas:,}")
                print(f"  Gas price: {self.w3.from_wei(self.gas_price, 'gwei'):.2f} Gwei")
                
                if self.eth_price:
                    eth_costs = [r.get('cost_eth', 0) for r in valid_results if 'cost_eth' in r]
                    usd_costs = [r.get('cost_usd', 0) for r in valid_results if 'cost_usd' in r]
                    
                    if eth_costs:
                        avg_cost_eth = np.mean(eth_costs)
                        avg_cost_usd = np.mean(usd_costs)
                        total_cost_eth = sum(eth_costs)
                        total_cost_usd = sum(usd_costs)
                        
                        print(f"\nCost Estimates:")
                        print(f"  Average cost per inference: {avg_cost_eth:.8f} ETH (${avg_cost_usd:.4f})")
                        print(f"  Total cost for all inferences: {total_cost_eth:.6f} ETH (${total_cost_usd:.2f})")
                        print(f"  Estimated cost for 1000 inferences: {avg_cost_eth*1000:.6f} ETH (${avg_cost_usd*1000:.2f})")
            
            confusion = np.zeros((10, 10), dtype=int)
            for r in valid_results:
                confusion[r['label']][r['prediction']] += 1
            
            print("\nConfusion Matrix:")
            print("    " + " ".join(f"{i:4d}" for i in range(10)))
            print("    " + "-" * 50)
            for i in range(10):
                print(f"{i} | " + " ".join(f"{confusion[i][j]:4d}" for j in range(10)))
            
            self.visualize_results(valid_results)
            
            if gas_values and not self.skip_gas:
                self.visualize_gas_usage(valid_results)
            
            return {
                'accuracy': accuracy,
                'correct_count': correct_count,
                'total_tested': len(valid_results),
                'errors': error_count,
                'time_taken': elapsed_time,
                'confusion_matrix': confusion.tolist(),
                'results': valid_results
            }
        else:
            print("No valid results obtained. Check for errors.")
            return None

def main():
    parser = argparse.ArgumentParser(description="Test MNIST model on-chain accuracy and gas usage")
    parser.add_argument("--contract", required=True, help="Contract address")
    parser.add_argument("--rpc", default="https://ethereum-holesky-rpc.publicnode.com", help="RPC URL") 
    parser.add_argument("--account", default=None, help="Account address for gas estimation (optional)")
    parser.add_argument("--samples", type=int, default=100, help="Number of test samples (default: 100)")
    parser.add_argument("--workers", type=int, default=5, help="Number of parallel workers (default: 5)")
    parser.add_argument("--no-price", action="store_true", help="Skip ETH price fetch")
    parser.add_argument("--skip-gas", action="store_true", help="Skip gas estimation (use when RPC has low limits)")
    
    args = parser.parse_args()
    
    try:
        tester = MNISTOnChainTester(
            args.contract, 
            args.rpc, 
            account_address=args.account,
            get_eth_price=not args.no_price,
            skip_gas=args.skip_gas
        )
        tester.evaluate_model(num_samples=args.samples, max_workers=args.workers)
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())   