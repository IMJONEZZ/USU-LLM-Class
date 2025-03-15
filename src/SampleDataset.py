class SampleDataset:
    def __init__(self):
        self.records = [
            # Instruction Chunks
            {
                "id": "pc_001_instr_0",
                "chunk_text": "If a team of 30 workers can build a house in 2 months,",
                "type": "instruction",
                "parent_id": "pc_001"
            },
            {
                "id": "pc_001_instr_1",
                "chunk_text": "how long will it take 300 workers to build the same house?",
                "type": "instruction",
                "parent_id": "pc_001"
            },
            # Response Chunks
            {
                "id": "pc_001_resp_0",
                "chunk_text": "It will still take 2 months,",
                "type": "response",
                "parent_id": "pc_001"
            },
            {
                "id": "pc_001_resp_1",
                "chunk_text": "assuming the work is not easily parallelizable.",
                "type": "response",
                "parent_id": "pc_001"
            },
            # Instruction Chunks
            {
                "id": "pc_002_instr_0",
                "chunk_text": "Which is heavier, a kilogram of cotton or a Euro?",
                "type": "instruction",
                "parent_id": "pc_002"
            },
            # Response Chunks
            {
                "id": "pc_002_resp_0",
                "chunk_text": "A kilogram of cotton is heavier in mass,",
                "type": "response",
                "parent_id": "pc_002"
            },
            {
                "id": "pc_002_resp_1",
                "chunk_text": "while a Euro is a unit of currency.",
                "type": "response",
                "parent_id": "pc_002"
            },
            # Adding 50 more instruction-response pairs

            # Pair 3
            {
                "id": "pc_003_instr_0",
                "chunk_text": "A girl wakes up and finds colorful lights in her living room,",
                "type": "instruction",
                "parent_id": "pc_003"
            },
            {
                "id": "pc_003_instr_1",
                "chunk_text": "with wrapped packages nearby. What day is it likely to be?",
                "type": "instruction",
                "parent_id": "pc_003"
            },
            {
                "id": "pc_003_resp_0",
                "chunk_text": "It is likely to be Christmas",
                "type": "response",
                "parent_id": "pc_003"
            },
            {
                "id": "pc_003_resp_1",
                "chunk_text": "or another gift-giving holiday.",
                "type": "response",
                "parent_id": "pc_003"
            },

            # Pair 4
            {
                "id": "pc_004_instr_0",
                "chunk_text": "What happens if you pop bubble wrap frequently?",
                "type": "instruction",
                "parent_id": "pc_004"
            },
            {
                "id": "pc_004_resp_0",
                "chunk_text": "Popping bubble wrap releases air from the bubbles.",
                "type": "response",
                "parent_id": "pc_004"
            },
            {
                "id": "pc_004_resp_1",
                "chunk_text": "It is not harmful but can be distracting to others.",
                "type": "response",
                "parent_id": "pc_004"
            },
        ]