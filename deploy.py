import json
import time
import sys
import os
from web3 import Web3
from eth_account import Account
from decimal import Decimal

class MNISTDeployer:
    def __init__(self, private_key, rpc_url):
        self.w3 = Web3(Web3.HTTPProvider(rpc_url))
        if not self.w3.is_connected():
            raise Exception(f"Failed to connect to blockchain via {rpc_url}")
        
        print(f"Connected to network: Chain ID {self.w3.eth.chain_id}")
        
        try:
            self.account = Account.from_key(private_key)
            self.address = self.account.address
            print(f"Using account: {self.address}")
        except Exception as e:
            raise Exception(f"Invalid private key: {e}")
        
        self.contract_address = None
        self.contract = None
        self.contract_abi = [
            {"inputs":[{"internalType":"int32","name":"_scaleFactor","type":"int32"},{"internalType":"uint16","name":"_w1ChunksCount","type":"uint16"},{"internalType":"uint16","name":"_w2ChunksCount","type":"uint16"},{"internalType":"uint16","name":"_w3ChunksCount","type":"uint16"}],"stateMutability":"nonpayable","type":"constructor"},
            {"anonymous":False,"inputs":[],"name":"BiasesInitialized","type":"event"},
            {"anonymous":False,"inputs":[{"indexed":False,"internalType":"uint8","name":"digit","type":"uint8"}],"name":"PredictionResult","type":"event"},
            {"anonymous":False,"inputs":[{"indexed":False,"internalType":"string","name":"weightType","type":"string"},{"indexed":False,"internalType":"uint16","name":"chunkIndex","type":"uint16"}],"name":"WeightsUploaded","type":"event"},
            {"inputs":[{"internalType":"uint8[400]","name":"inputImage","type":"uint8[400]"}],"name":"predict","outputs":[{"internalType":"uint8","name":"","type":"uint8"}],"stateMutability":"nonpayable","type":"function"},
            {"inputs":[],"name":"getUploadStatus","outputs":[{"internalType":"bool","name":"biasesInit","type":"bool"},{"internalType":"uint16","name":"w1Uploaded","type":"uint16"},{"internalType":"uint16","name":"w1Total","type":"uint16"},{"internalType":"uint16","name":"w2Uploaded","type":"uint16"},{"internalType":"uint16","name":"w2Total","type":"uint16"},{"internalType":"uint16","name":"w3Uploaded","type":"uint16"},{"internalType":"uint16","name":"w3Total","type":"uint16"},{"internalType":"bool","name":"ready","type":"bool"}],"stateMutability":"view","type":"function"},
            {"inputs":[],"name":"isModelReady","outputs":[{"internalType":"bool","name":"","type":"bool"}],"stateMutability":"view","type":"function"},
            {"inputs":[{"internalType":"int8[32]","name":"_hidden1Biases","type":"int8[32]"},{"internalType":"int8[16]","name":"_hidden2Biases","type":"int8[16]"},{"internalType":"int8[10]","name":"_outputBiases","type":"int8[10]"}],"name":"initializeBiases","outputs":[],"stateMutability":"nonpayable","type":"function"},
            {"inputs":[{"internalType":"uint16","name":"chunkIndex","type":"uint16"},{"internalType":"int8[]","name":"weights","type":"int8[]"},{"internalType":"uint32","name":"startPos","type":"uint32"}],"name":"uploadInputHidden1Weights","outputs":[],"stateMutability":"nonpayable","type":"function"},
            {"inputs":[{"internalType":"uint16","name":"chunkIndex","type":"uint16"},{"internalType":"int8[]","name":"weights","type":"int8[]"},{"internalType":"uint32","name":"startPos","type":"uint32"}],"name":"uploadHidden1Hidden2Weights","outputs":[],"stateMutability":"nonpayable","type":"function"},
            {"inputs":[{"internalType":"uint16","name":"chunkIndex","type":"uint16"},{"internalType":"int8[]","name":"weights","type":"int8[]"},{"internalType":"uint32","name":"startPos","type":"uint32"}],"name":"uploadHidden2OutputWeights","outputs":[],"stateMutability":"nonpayable","type":"function"}
        ]
        self.eth_price = Decimal('3000.0')
        print(f"Current ETH price: ${self.eth_price:.2f}")
        
        self.min_gas_price = 5
        
        self.model_data = self.load_model_data()
    
    def load_model_data(self, json_file=None):
        if json_file is None:
            # Try to find model file in standard locations
            possible_paths = [
                'quantized_model_mini.json',
                '../models/quantized_model_mini.json',
                'models/quantized_model_mini.json'
            ]
            
            for path in possible_paths:
                if os.path.exists(path):
                    json_file = path
                    break
            
            if json_file is None:
                raise Exception("Model file not found in standard locations")
        
        try:
            with open(json_file, 'r') as f:
                model_data = json.load(f)
                
                required_keys = ['w1_chunks', 'w2_chunks', 'w3_chunks', 'b1', 'b2', 'b3', 'scale_factor']
                for key in required_keys:
                    if key not in model_data:
                        raise Exception(f"Missing '{key}' in model data")
                
                print(f"Loaded model data from {json_file}:")
                print(f"- Scale factor: {model_data['scale_factor']}")
                print(f"- Input size: {model_data.get('input_size', 400)}")
                print(f"- Hidden layer 1: {model_data.get('hidden1_size', 32)} neurons")
                print(f"- Hidden layer 2: {model_data.get('hidden2_size', 16)} neurons")
                print(f"- Output size: {model_data.get('output_size', 10)}")
                print(f"- Weight chunks: {len(model_data['w1_chunks'])} w1, {len(model_data['w2_chunks'])} w2, {len(model_data['w3_chunks'])} w3")
                
                return model_data
        except Exception as e:
            raise Exception(f"Error loading model data: {e}")
    
    def get_gas_price(self):
        try:
            gas_price = self.w3.eth.gas_price
            
            min_price_wei = self.w3.to_wei(self.min_gas_price, 'gwei')
            if gas_price < min_price_wei:
                print(f"Network gas price too low ({self.w3.from_wei(gas_price, 'gwei'):.2f} gwei), using minimum: {self.min_gas_price} gwei")
                gas_price = min_price_wei
                
            return gas_price
        except Exception as e:
            print(f"Error getting gas price: {e}")
            return self.w3.to_wei(self.min_gas_price, 'gwei')
    
    def estimate_gas(self, func, *args):
        try:
            if self.contract is None:
                raise Exception("Contract not initialized")
                
            contract_func = getattr(self.contract.functions, func)
            gas_estimate = contract_func(*args).estimate_gas({'from': self.address})
            buffered_estimate = gas_estimate + 50000
            return buffered_estimate
        except Exception as e:
            print(f"Gas estimation error for {func}: {e}")
            safe_defaults = {
                'initializeBiases': 2000000,
                'uploadInputHidden1Weights': 36000000,
                'uploadHidden1Hidden2Weights': 3000000,
                'uploadHidden2OutputWeights': 2000000
            }
            return safe_defaults.get(func, 5000000)
    
    def calculate_cost(self, gas_estimate):
        gas_price = self.get_gas_price()
        
        gas_price_ether = Decimal(str(self.w3.from_wei(gas_price, 'ether')))
        gas_estimate_dec = Decimal(str(gas_estimate))
        
        eth_cost = gas_price_ether * gas_estimate_dec
        usd_cost = eth_cost * self.eth_price
        
        return eth_cost, usd_cost, gas_price
    
    def confirm_transaction(self, func_name, eth_cost, usd_cost, gas_price, gas_estimate):
        print("\n----- Transaction Details -----")
        print(f"Function: {func_name}")
        print(f"Gas Price: {self.w3.from_wei(gas_price, 'gwei'):.2f} gwei")
        print(f"Gas Estimate: {gas_estimate}")
        print(f"Estimated cost: {eth_cost:.6f} ETH")
        print(f"Estimated cost in USD: ${usd_cost:.2f}")
        print(f"Network: Chain ID {self.w3.eth.chain_id}")
        print("------------------------------")
        
        confirm = input("Continue with this transaction? (y/n): ").strip().lower()
        return confirm == 'y' or confirm == 'yes'
    
    def get_transaction_hash_from_error(self, error_str):
        import re
        matches = re.findall(r'0x[a-fA-F0-9]{64}', error_str)
        if matches:
            return matches[0]
        return None
    
    def check_recent_pending_tx(self, func_name):
        try:
            pending_count = 0
            for tx_hash in self.w3.eth.get_block('pending').transactions:
                try:
                    tx = self.w3.eth.get_transaction(tx_hash)
                    if tx and tx['from'].lower() == self.address.lower():
                        pending_count += 1
                        print(f"Found pending transaction: {tx_hash.hex()}")
                        
                        if pending_count > 10:
                            print("Too many pending transactions. Please wait for them to confirm.")
                            return None
                except Exception:
                    continue
            return None
        except Exception:
            return None
    
    def wait_for_receipt(self, tx_hash, max_retries=30, retry_delay=5):
        print(f"Waiting for transaction confirmation... ({tx_hash.hex()})")
        
        for attempt in range(max_retries):
            try:
                receipt = self.w3.eth.get_transaction_receipt(tx_hash)
                if receipt is not None:
                    return receipt
            except Exception as e:
                print(f"Attempt {attempt+1}/{max_retries}: Error retrieving receipt - {str(e)}")
                
            time.sleep(retry_delay)
            print(f"Still waiting for confirmation... (attempt {attempt+1}/{max_retries})")
            
            if self.contract is not None:
                try:
                    status = self.contract.functions.getUploadStatus().call()
                    print("Current contract status while waiting:")
                    print(f"  Biases: {'✓' if status[0] else '✗'}")
                    print(f"  W1 Chunks: {status[1]}/{status[2]}")
                    print(f"  W2 Chunks: {status[3]}/{status[4]}")
                    print(f"  W3 Chunks: {status[5]}/{status[6]}")
                    print(f"  Model Ready: {'✓' if status[7] else '✗'}")
                except Exception:
                    pass
        
        print(f"Timed out waiting for receipt after {max_retries*retry_delay} seconds")
        
        try:
            tx = self.w3.eth.get_transaction(tx_hash)
            if tx and tx.get('blockNumber'):
                print(f"Transaction was mined in block {tx['blockNumber']}, but receipt unavailable")
                return {'status': 1, 'blockNumber': tx['blockNumber'], 'gasUsed': 0}
        except Exception:
            pass
            
        return None
        
    def verify_transaction(self, tx_hash):
        print(f"Checking transaction status for {tx_hash.hex()}...")
        
        receipt = self.wait_for_receipt(tx_hash)
        if receipt:
            status = receipt.get('status')
            if status == 1:
                return True, receipt
            else:
                print(f"⚠️ Transaction failed with status: {status}")
                return False, receipt
        
        print("⚠️ Transaction may still be processing...")
        return None, None

    def send_transaction(self, func, *args):
        try:
            if self.contract is None and func != 'contract_deployment':
                raise Exception("Contract not initialized")
                
            if func == 'contract_deployment':
                # Special handling for contract deployment
                bytecode = args[0]
                constructor_args = args[1:]
                
                gas_limit = 6000000  # Set a high limit for contract deployment
                
                eth_cost, usd_cost, gas_price = self.calculate_cost(gas_limit)
                
                if not self.confirm_transaction('contract_deployment', eth_cost, usd_cost, gas_price, gas_limit):
                    print("Contract deployment cancelled by user")
                    return None
                
                nonce = self.w3.eth.get_transaction_count(self.address)
                
                contract = self.w3.eth.contract(abi=self.contract_abi, bytecode=bytecode)
                
                tx = contract.constructor(*constructor_args).build_transaction({
                    'chainId': self.w3.eth.chain_id,
                    'gas': gas_limit,
                    'gasPrice': gas_price,
                    'nonce': nonce,
                    'from': self.address
                })
            else:
                # Regular function call
                gas_limit = self.estimate_gas(func, *args)
                
                eth_cost, usd_cost, gas_price = self.calculate_cost(gas_limit)
                
                if not self.confirm_transaction(func, eth_cost, usd_cost, gas_price, gas_limit):
                    print("Transaction cancelled by user")
                    return None
                
                nonce = self.w3.eth.get_transaction_count(self.address)
                
                contract_func = getattr(self.contract.functions, func)
                tx = contract_func(*args).build_transaction({
                    'chainId': self.w3.eth.chain_id,
                    'gas': gas_limit,
                    'gasPrice': gas_price,
                    'nonce': nonce,
                    'from': self.address
                })
            
            print(f"Transaction built successfully. Signing with account {self.address}...")
            
            signed_tx = self.w3.eth.account.sign_transaction(tx, private_key=self.account.key)
            
            print("Transaction signed successfully.")
            
            # Extract raw transaction
            if hasattr(signed_tx, 'rawTransaction'):
                raw_tx = signed_tx.rawTransaction
            elif hasattr(signed_tx, 'raw_transaction'):
                raw_tx = signed_tx.raw_transaction
            else:
                raw_tx = signed_tx.get('rawTransaction', signed_tx.get('raw_transaction'))
                if not raw_tx:
                    print("Warning: Unusual SignedTransaction object format. Attempting manual extraction...")
                    print(f"Signed transaction object type: {type(signed_tx)}")
                    print(f"Signed transaction attributes: {dir(signed_tx)}")
                    if isinstance(signed_tx, dict) and 'rawTransaction' in signed_tx:
                        raw_tx = signed_tx['rawTransaction']
                    else:
                        raise Exception("Could not extract raw transaction data from signed transaction")
            
            try:
                tx_hash = self.w3.eth.send_raw_transaction(raw_tx)
                tx_hash_hex = tx_hash.hex()
                print(f"Transaction sent: {tx_hash_hex}")
            except Exception as e:
                error_str = str(e)
                print(f"Error sending transaction: {error_str}")
                
                if "already known" in error_str or "nonce too low" in error_str:
                    print("Transaction already submitted! Checking for pending transactions...")
                    
                    tx_hash_from_error = self.get_transaction_hash_from_error(error_str)
                    if tx_hash_from_error:
                        tx_hash = Web3.to_bytes(hexstr=tx_hash_from_error)
                        print(f"Found transaction hash from error: {tx_hash_from_error}")
                    else:
                        if func == 'contract_deployment':
                            print("Could not determine contract deployment transaction hash.")
                            return None
                        
                        print("Checking status to verify if action was already completed...")
                        try:
                            status = self.contract.functions.getUploadStatus().call()
                            
                            if func == 'initializeBiases' and status[0]:
                                print("✓ Biases already initialized!")
                                return {'status': 1, 'already_done': True}
                            elif 'upload' in func.lower():
                                # More complex handling of chunk upload status
                                is_completed = False
                                
                                if 'InputHidden1' in func:
                                    chunk_index = args[0]
                                    if chunk_index < status[1]:
                                        print(f"✓ W1 Chunk {chunk_index} already uploaded!")
                                        is_completed = True
                                elif 'Hidden1Hidden2' in func:
                                    chunk_index = args[0]
                                    if chunk_index < status[3]:
                                        print(f"✓ W2 Chunk {chunk_index} already uploaded!")
                                        is_completed = True
                                elif 'Hidden2Output' in func:
                                    chunk_index = args[0]
                                    if chunk_index < status[5]:
                                        print(f"✓ W3 Chunk {chunk_index} already uploaded!")
                                        is_completed = True
                                
                                if is_completed:
                                    return {'status': 1, 'already_done': True}
                        except Exception as status_err:
                            print(f"Error checking status: {status_err}")
                        
                        print("Could not determine transaction hash. Waiting to see if state changes...")
                        time.sleep(10)
                        
                        try:
                            new_status = self.contract.functions.getUploadStatus().call()
                            if new_status != status:
                                print("✓ Contract state has changed, transaction likely succeeded!")
                                return {'status': 1, 'already_done': True}
                        except Exception:
                            pass
                            
                        # Try another approach - look for pending transactions
                        tx_hash = self.check_recent_pending_tx(func)
                        if not tx_hash:
                            print("No transaction hash found. Please check manually.")
                            return None
                else:
                    print("Error not related to duplicate transaction. Aborting.")
                    return None
            
            success, receipt = self.verify_transaction(tx_hash)
            
            if success is None:
                print("Transaction status uncertain. Please check manually with tx hash.")
                return {'status': 1, 'tx_hash': tx_hash.hex(), 'uncertain': True}
                
            if not success:
                print("Transaction was mined but failed.")
                return None
                
            actual_gas_used = receipt['gasUsed']
            actual_eth_cost = Decimal(str(self.w3.from_wei(actual_gas_used * gas_price, 'ether')))
            actual_usd_cost = actual_eth_cost * self.eth_price
            
            print(f"Transaction confirmed! Block: {receipt['blockNumber']}")
            print(f"Gas used: {actual_gas_used} ({(actual_gas_used/gas_limit)*100:.1f}% of estimate)")
            print(f"Actual cost: {actual_eth_cost:.6f} ETH")
            print(f"Actual cost in USD: ${actual_usd_cost:.2f}")
            
            # For contract deployment, extract contract address and initialize contract
            if func == 'contract_deployment' and 'contractAddress' in receipt:
                self.contract_address = receipt['contractAddress']
                self.contract = self.w3.eth.contract(address=self.contract_address, abi=self.contract_abi)
                print(f"Contract deployed at address: {self.contract_address}")
                
                # Save contract address to file for future use
                with open("contract_address.txt", "w") as f:
                    f.write(self.contract_address)
                print("Contract address saved to contract_address.txt")
            
            return receipt
            
        except Exception as e:
            print(f"Transaction failed: {str(e)}")
            import traceback
            traceback.print_exc()
            return None

    def deploy_contract(self):
        print("\n====== Deploying MNIST Model Contract (20x20 input) ======")
        
        try:
            # Try multiple bytecode file locations
            bytecode_locations = ["./bytecode.txt", "bytecode.txt", "../contracts/bytecode.txt", "../bytecode.txt"]
            bytecode = None
            
            for loc in bytecode_locations:
                try:
                    if os.path.exists(loc):
                        with open(loc, "r") as f:
                            bytecode = f.read().strip()
                            if not bytecode.startswith("0x"):
                                bytecode = "0x" + bytecode
                            print(f"Read {len(bytecode)-2} bytes of bytecode from {loc}")
                            break
                except:
                    continue
            
            if bytecode is None:
                raise Exception("Bytecode file not found. Please compile the contract first.")
            
            # Prepare constructor arguments
            scale_factor = int(self.model_data['scale_factor'])
            w1_chunks_count = len(self.model_data['w1_chunks'])
            w2_chunks_count = len(self.model_data['w2_chunks'])
            w3_chunks_count = len(self.model_data['w3_chunks'])
            
            print("\nConstructor arguments:")
            print(f"  Scale factor: {scale_factor}")
            print(f"  W1 chunks: {w1_chunks_count} (400x32 matrix)")
            print(f"  W2 chunks: {w2_chunks_count} (32x16 matrix)")
            print(f"  W3 chunks: {w3_chunks_count} (16x10 matrix)")
            
            # Deploy the contract
            receipt = self.send_transaction('contract_deployment', bytecode, scale_factor, w1_chunks_count, w2_chunks_count, w3_chunks_count)
            
            if receipt is None:
                print("❌ Contract deployment failed")
                return False
                
            if 'contractAddress' not in receipt:
                print("❌ Contract deployment succeeded but contract address not found in receipt")
                return False
                
            print(f"✅ Contract deployed successfully at {self.contract_address}")
            return True
            
        except Exception as e:
            print(f"❌ Contract deployment error: {e}")
            return False
    
    def initialize_contract_from_address(self, contract_address):
        try:
            self.contract_address = Web3.to_checksum_address(contract_address)
            self.contract = self.w3.eth.contract(address=self.contract_address, abi=self.contract_abi)
            
            # Verify connection
            try:
                ready = self.contract.functions.isModelReady().call()
                status = self.contract.functions.getUploadStatus().call()
                print(f"Connected to existing contract at {self.contract_address}")
                print(f"Contract status: Ready = {'✓' if ready else '✗'}")
                print(f"  Biases: {'✓' if status[0] else '✗'}")
                print(f"  W1 Chunks: {status[1]}/{status[2]}")
                print(f"  W2 Chunks: {status[3]}/{status[4]}")
                print(f"  W3 Chunks: {status[5]}/{status[6]}")
                
                return True
            except Exception as e:
                print(f"Error connecting to contract: {e}")
                return False
                
        except Exception as e:
            print(f"Error initializing contract: {e}")
            return False
    
    def initialize_biases(self):
        if self.contract is None:
            print("⚠️ Contract not initialized")
            return False
            
        status = self.contract.functions.getUploadStatus().call()
        if status[0]:
            print("Biases already initialized")
            return True
        
        print("\n--- Initializing Biases ---")
        hidden1_biases = self.model_data['b1']
        hidden2_biases = self.model_data['b2']
        output_biases = self.model_data['b3']
        
        print(f"Hidden1 biases: {len(hidden1_biases)} values")
        print(f"Hidden2 biases: {len(hidden2_biases)} values")
        print(f"Output biases: {len(output_biases)} values")
        
        receipt = self.send_transaction('initializeBiases', hidden1_biases, hidden2_biases, output_biases)
        if receipt is None:
            return False
        
        if receipt.get('already_done', False):
            return True
            
        if receipt.get('uncertain', False):
            print("Transaction sent, checking if biases were initialized...")
            time.sleep(10)
            
        try:
            new_status = self.contract.functions.getUploadStatus().call()
            if new_status[0]:
                print("✓ Biases successfully initialized")
                return True
            else:
                print("⚠️ Transaction completed but biases not initialized")
                return False
        except Exception as e:
            print(f"Error checking biases status: {e}")
            return False
    
    def upload_weights(self):
        if self.contract is None:
            print("⚠️ Contract not initialized")
            return False
            
        status = self.contract.functions.getUploadStatus().call()
        
        if not status[0]:
            print("⚠️ Biases must be initialized first")
            return False
        
        # Upload w1 chunks (input->hidden1)
        status = self.contract.functions.getUploadStatus().call()
        w1_total = status[2]
        w1_uploaded = status[1]
            
        print(f"\nUploading W1 weights ({len(self.model_data['w1_chunks'])} chunks)")
        for i in range(w1_uploaded, w1_total):
            chunk = self.model_data['w1_chunks'][i]
            start_pos = i * len(chunk)
            
            print(f"--- Uploading W1 chunk {i+1}/{w1_total} ({len(chunk)} weights, start pos: {start_pos}) ---")
            receipt = self.send_transaction('uploadInputHidden1Weights', i, chunk, start_pos)
            
            if receipt is None:
                return False
            
            if receipt.get('already_done', False):
                continue
                
            if receipt.get('uncertain', False):
                print(f"Transaction sent, checking if W1 chunk {i} was uploaded...")
                time.sleep(10)
            
            try:
                new_status = self.contract.functions.getUploadStatus().call()
                if new_status[1] > w1_uploaded:
                    print(f"✓ W1 chunk {i} successfully uploaded")
                    w1_uploaded = new_status[1]
                else:
                    print(f"⚠️ Transaction completed but W1 chunk {i} not recorded")
                    return False
            except Exception as e:
                print(f"Error checking W1 chunk status: {e}")
                return False
                
            time.sleep(2)  # Small delay between transactions
        
        # Upload w2 chunks (hidden1->hidden2)
        status = self.contract.functions.getUploadStatus().call()
        w2_total = status[4]
        w2_uploaded = status[3]
        
        for i in range(w2_uploaded, w2_total):
            chunk = self.model_data['w2_chunks'][i]
            start_pos = i * len(chunk)
            
            print(f"\n--- Uploading W2 chunk {i+1}/{w2_total} ({len(chunk)} weights, start pos: {start_pos}) ---")
            receipt = self.send_transaction('uploadHidden1Hidden2Weights', i, chunk, start_pos)
            
            if receipt is None:
                return False
            
            if receipt.get('already_done', False):
                continue
                
            if receipt.get('uncertain', False):
                print(f"Transaction sent, checking if W2 chunk {i} was uploaded...")
                time.sleep(10)
            
            try:
                new_status = self.contract.functions.getUploadStatus().call()
                if new_status[3] > w2_uploaded:
                    print(f"✓ W2 chunk {i} successfully uploaded")
                    w2_uploaded = new_status[3]
                else:
                    print(f"⚠️ Transaction completed but W2 chunk {i} not recorded")
                    return False
            except Exception as e:
                print(f"Error checking W2 chunk status: {e}")
                return False
                
            time.sleep(2)  # Small delay between transactions
        
        # Upload w3 chunks (hidden2->output)
        status = self.contract.functions.getUploadStatus().call()
        w3_total = status[6]
        w3_uploaded = status[5]
        
        for i in range(w3_uploaded, w3_total):
            chunk = self.model_data['w3_chunks'][i]
            start_pos = i * len(chunk)
            
            print(f"\n--- Uploading W3 chunk {i+1}/{w3_total} ({len(chunk)} weights, start pos: {start_pos}) ---")
            receipt = self.send_transaction('uploadHidden2OutputWeights', i, chunk, start_pos)
            
            if receipt is None:
                return False
            
            if receipt.get('already_done', False):
                continue
                
            if receipt.get('uncertain', False):
                print(f"Transaction sent, checking if W3 chunk {i} was uploaded...")
                time.sleep(10)
            
            try:
                new_status = self.contract.functions.getUploadStatus().call()
                if new_status[5] > w3_uploaded:
                    print(f"✓ W3 chunk {i} successfully uploaded")
                    w3_uploaded = new_status[5]
                else:
                    print(f"⚠️ Transaction completed but W3 chunk {i} not recorded")
                    return False
            except Exception as e:
                print(f"Error checking W3 chunk status: {e}")
                return False
                
            time.sleep(2)  # Small delay between transactions
        
        # Final check
        try:
            is_ready = self.contract.functions.isModelReady().call()
            if is_ready:
                print("\n✅ All weights uploaded successfully!")
                return True
            else:
                status = self.contract.functions.getUploadStatus().call()
                print("\n⚠️ All weights uploaded but model not ready. Current status:")
                print(f"  Biases: {'✓' if status[0] else '✗'}")
                print(f"  W1 Chunks: {status[1]}/{status[2]}")
                print(f"  W2 Chunks: {status[3]}/{status[4]}")
                print(f"  W3 Chunks: {status[5]}/{status[6]}")
                print(f"  Model Ready: {'✓' if status[7] else '✗'}")
                return False
        except Exception as e:
            print(f"Error checking final model status: {e}")
            return False
    
    def deploy_model(self):
        print("\n====== MNIST Model Deployment ======")
        
        # Step 1: Check if we need to deploy a new contract or use existing one
        if os.path.exists("contract_address.txt"):
            with open("contract_address.txt", "r") as f:
                saved_address = f.read().strip()
                if saved_address:
                    print(f"Found saved contract address: {saved_address}")
                    use_existing = input("Use this existing contract? (y/n): ").strip().lower()
                    if use_existing == 'y' or use_existing == 'yes':
                        if self.initialize_contract_from_address(saved_address):
                            print("✅ Using existing contract")
                        else:
                            print("❌ Could not connect to existing contract")
                            if input("Deploy new contract instead? (y/n): ").strip().lower() in ['y', 'yes']:
                                if not self.deploy_contract():
                                    return False
                            else:
                                return False
                    else:
                        if not self.deploy_contract():
                            return False
                else:
                    if not self.deploy_contract():
                        return False
        else:
            if not self.deploy_contract():
                return False
        
        # Step 2: Check contract status
        try:
            status = self.contract.functions.getUploadStatus().call()
            is_ready = self.contract.functions.isModelReady().call()
            
            print("\nCurrent deployment status:")
            print(f"  Biases: {'✓' if status[0] else '✗'}")
            print(f"  W1 Chunks: {status[1]}/{status[2]}")
            print(f"  W2 Chunks: {status[3]}/{status[4]}")
            print(f"  W3 Chunks: {status[5]}/{status[6]}")
            print(f"  Model Ready: {'✓' if is_ready else '✗'}")
            
            if is_ready:
                print("\n🎉 Model is already fully deployed and ready for use! 🎉")
                return True
                
        except Exception as e:
            print(f"Error checking contract status: {e}")
            return False
        
        # Step 3: Estimate deployment costs
        print("\nEstimating remaining deployment costs...")
        
        total_gas = 0
        remaining_steps = []
        
        # Add biases initialization if needed
        if not status[0]:
            try:
                gas = self.estimate_gas('initializeBiases', 
                                       self.model_data['b1'], 
                                       self.model_data['b2'], 
                                       self.model_data['b3'])
                total_gas += gas
                remaining_steps.append(("Biases", gas))
                print(f"  Biases: {gas} gas")
            except Exception as e:
                gas = 200000
                total_gas += gas
                remaining_steps.append(("Biases", gas))
                print(f"  Biases: {gas} gas (fallback estimate)")
        
        # Add W1 chunks if needed
        w1_uploaded = status[1]
        w1_total = status[2]
        for i in range(w1_uploaded, w1_total):
            try:
                chunk = self.model_data['w1_chunks'][i]
                start_pos = i * len(chunk)
                if status[0]:  # Only try to estimate if biases are initialized
                    gas = self.estimate_gas('uploadInputHidden1Weights', i, chunk, start_pos)
                else:
                    gas = 500000
                total_gas += gas
                remaining_steps.append((f"W1 Chunk {i}", gas))
                print(f"  W1 Chunk {i}: {gas} gas")
            except Exception as e:
                gas = 500000
                total_gas += gas
                remaining_steps.append((f"W1 Chunk {i}", gas))
                print(f"  W1 Chunk {i}: {gas} gas (fallback estimate)")
        
        # Add W2 chunks if needed
        w2_uploaded = status[3]
        w2_total = status[4]
        for i in range(w2_uploaded, w2_total):
            try:
                chunk = self.model_data['w2_chunks'][i]
                start_pos = i * len(chunk)
                if status[0]:  # Only try to estimate if biases are initialized
                    gas = self.estimate_gas('uploadHidden1Hidden2Weights', i, chunk, start_pos)
                else:
                    gas = 300000
                total_gas += gas
                remaining_steps.append((f"W2 Chunk {i}", gas))
                print(f"  W2 Chunk {i}: {gas} gas")
            except Exception as e:
                gas = 300000
                total_gas += gas
                remaining_steps.append((f"W2 Chunk {i}", gas))
                print(f"  W2 Chunk {i}: {gas} gas (fallback estimate)")
        
        # Add W3 chunks if needed
        w3_uploaded = status[5]
        w3_total = status[6]
        for i in range(w3_uploaded, w3_total):
            try:
                chunk = self.model_data['w3_chunks'][i]
                start_pos = i * len(chunk)
                if status[0]:  # Only try to estimate if biases are initialized
                    gas = self.estimate_gas('uploadHidden2OutputWeights', i, chunk, start_pos)
                else:
                    gas = 200000
                total_gas += gas
                remaining_steps.append((f"W3 Chunk {i}", gas))
                print(f"  W3 Chunk {i}: {gas} gas")
            except Exception as e:
                gas = 200000
                total_gas += gas
                remaining_steps.append((f"W3 Chunk {i}", gas))
                print(f"  W3 Chunk {i}: {gas} gas (fallback estimate)")
        
        if not remaining_steps:
            print("  No remaining steps needed!")
            return True
        
        eth_cost, usd_cost, gas_price = self.calculate_cost(total_gas)
        
        print(f"\nTotal estimated deployment cost: {eth_cost:.6f} ETH")
        print(f"Total estimated cost in USD: ${usd_cost:.2f}")
        print(f"Gas Price: {self.w3.from_wei(gas_price, 'gwei'):.2f} gwei")
        
        print("\nWarning: This will execute multiple transactions that must be confirmed individually.")
        confirm = input("Begin deployment process? (y/n): ").strip().lower()
        if not (confirm == 'y' or confirm == 'yes'):
            print("Deployment cancelled by user")
            return False
        
        try:
            # Initialize biases first
            if not status[0]:
                max_attempts = 3
                for attempt in range(max_attempts):
                    init_success = self.initialize_biases()
                    if init_success:
                        break
                    elif attempt < max_attempts - 1:
                        print(f"Biases initialization attempt {attempt+1} failed. Retrying...")
                        time.sleep(5)
                    else:
                        print("❌ Biases initialization failed after multiple attempts. Deployment aborted.")
                        return False
            
            # Upload weights
            max_attempts = 3
            for attempt in range(max_attempts):
                upload_success = self.upload_weights()
                if upload_success:
                    break
                elif attempt < max_attempts - 1:
                    print(f"Weight upload attempt {attempt+1} failed. Retrying...")
                    time.sleep(5)
                else:
                    print("❌ Weight upload failed after multiple attempts. Deployment incomplete.")
                    return False
            
            # Final check
            is_ready = self.contract.functions.isModelReady().call()
            if is_ready:
                print("\n🎉 Model successfully deployed and ready for inference! 🎉")
                print(f"Contract address: {self.contract_address}")
                return True
            else:
                print("\n⚠️ All steps completed but model not marked as ready. Please check status.")
                status = self.contract.functions.getUploadStatus().call()
                print("\nCurrent deployment status:")
                print(f"  Biases: {'✓' if status[0] else '✗'}")
                print(f"  W1 Chunks: {status[1]}/{status[2]}")
                print(f"  W2 Chunks: {status[3]}/{status[4]}")
                print(f"  W3 Chunks: {status[5]}/{status[6]}")
                print(f"  Model Ready: {'✓' if is_ready else '✗'}")
                return False
                
        except Exception as e:
            print(f"\n❌ Deployment error: {e}")
            print("\nCurrent deployment status:")
            try:
                status = self.contract.functions.getUploadStatus().call()
                print(f"  Biases: {'✓' if status[0] else '✗'}")
                print(f"  W1 Chunks: {status[1]}/{status[2]}")
                print(f"  W2 Chunks: {status[3]}/{status[4]}")
                print(f"  W3 Chunks: {status[5]}/{status[6]}")
                print(f"  Model Ready: {'✓' if self.contract.functions.isModelReady().call() else '✗'}")
            except:
                print("  Unable to retrieve status")
            
            return False

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Deploy MNIST model to blockchain")
    parser.add_argument("--contract", help="Existing contract address (optional)")
    parser.add_argument("--key", required=True, help="Private key")
    parser.add_argument("--rpc", default="https://ethereum-holesky-rpc.publicnode.com", help="RPC URL")
    parser.add_argument("--action", choices=["deploy", "status"], default="deploy", help="Action to perform")
    parser.add_argument("--model", help="Model data JSON file path")
    parser.add_argument("--gas-price", type=float, default=5.0, help="Minimum gas price in gwei")
    
    args = parser.parse_args()
    
    try:
        try:
            import pkg_resources
            web3_version = pkg_resources.get_distribution("web3").version
            print(f"Web3.py version: {web3_version}")
        except:
            print("Could not determine Web3.py version")
        
        deployer = MNISTDeployer(args.key, args.rpc)
        deployer.min_gas_price = args.gas_price
        
        if args.action == "deploy":
            if args.contract:
                deployer.initialize_contract_from_address(args.contract)
            deployer.deploy_model()
        elif args.action == "status":
            if args.contract:
                if deployer.initialize_contract_from_address(args.contract):
                    try:
                        status = deployer.contract.functions.getUploadStatus().call()
                        is_ready = deployer.contract.functions.isModelReady().call()
                        
                        print("\nCurrent deployment status:")
                        print(f"  Biases: {'✓' if status[0] else '✗'}")
                        print(f"  W1 Chunks: {status[1]}/{status[2]}")
                        print(f"  W2 Chunks: {status[3]}/{status[4]}")
                        print(f"  W3 Chunks: {status[5]}/{status[6]}")
                        print(f"  Model Ready: {'✓' if is_ready else '✗'}")
                        
                        if is_ready:
                            print("\nModel is fully deployed and ready for inference!")
                        else:
                            print("\nModel deployment is incomplete.")
                            
                            missing_components = []
                            if not status[0]:
                                missing_components.append("Biases")
                            
                            if status[1] < status[2]:
                                missing_components.append(f"W1 Chunks ({status[1]}/{status[2]} uploaded)")
                            
                            if status[3] < status[4]:
                                missing_components.append(f"W2 Chunks ({status[3]}/{status[4]} uploaded)")
                            
                            if status[5] < status[6]:
                                missing_components.append(f"W3 Chunks ({status[5]}/{status[6]} uploaded)")
                            
                            if missing_components:
                                print(f"Missing components: {', '.join(missing_components)}")
                                print("\nRun the deploy command to complete the deployment:")
                                print(f"python deploy.py --contract {args.contract} --key YOUR_KEY --action deploy")
                    except Exception as e:
                        print(f"Error checking status: {e}")
            else:
                print("Error: --contract address is required for status action")
                sys.exit(1)
    
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()