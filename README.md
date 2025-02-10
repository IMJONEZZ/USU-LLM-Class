# How my code works? 

The data loder is under the folder HW3

2. SWDataset Class: Stores a dataset (self.data) as a list of dictionaries. Implements __len__ to return the dataset length. Implements __getitem__ to return a single data sample by index.  

3. load_data step: Opens "SW_EpisodeIV_VI.json" and reads it into a Python dictionary. Initializes an SWDataset instance with the data. Wraps SWDataset in a PyTorch DataLoader with batch_size = 2 and shuffle = True. Returns the DataLoader in a dictionary for easy pipeline integration.

4. ZenML Pipeline (data_process_pipeline): Defines a step (load_data()) and connects it into a pipeline. When executed, it triggers load_data(), ensuring modularity. 

# Why this approach? 

1. Modular Design: SWDataset is reusable and works seamlessly with PyTorch.

2. Scalability: Using DataLoader enables efficient batch processing.

3. Pipeline Integration: ZenML ensures flexibility and reproducibility in workflows

4. Shuffling: Ensures model training (in the future) generalizes well.

# Why I Wrote These Tests? 

1. Ensures dataset length is correctly reported. 

2. Checks individual item retrieval from SWDataset. 

3. Verifies that load_data loads data correctly and returns a DataLoader. 

4. Ensures the entire ZenML pipeline runs without errors. 

5. Checks that DataLoader correctly batches data.

6. 	Ensures shuffling works by comparing two iterations.



