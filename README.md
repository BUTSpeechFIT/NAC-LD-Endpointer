# NAC-LD-Endpointer
Codebase for the work "Streaming Endpointer for Spoken Dialogue using Neural Audio Codecs and Label-Delayed Training", accepted at ASRU 2025
___

## Data Processing Pipeline
```
┌─────────────────┐
│  Raw Dataset(s) │
└────────┬────────┘
         │
         ▼
┌─────────────────────────────────────┐
│  Standardize to Common JSON Format  │
└────────┬────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────┐
│   VAD Processing (Silero VAD)       │
│  • Trim turn-level beg/end silence  │
└────────┬────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────┐
│   Add Turn Annotations              │
│  • Add <user_end>/<system_end>      │
│  • Handle timing conflicts          │
└────────┬────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────┐
│   Processed Training Data           │
│  • Special tokens inserted          │
│  • Ready for model training         │
└─────────────────────────────────────┘
```

### Processing Details

Each step produces intermediate outputs that can be cached:

| Step | Output Path | Override Flag |
|------|-------------|---------------|
| Standardization | `preprocessed_{mode}.json` | `override_preprocessed_data` |
| VAD Processing | `vad_processed_{mode}/` | `override_vad_data` |
| Turn Annotations | `processed_{mode}/` | `override_processed_data` |
| Length Filtering | `filtered_{mode}/` | `override_filtered_data` |

The length filtered files are used for training. 

**Example processed segment:**
```json
{
  "audio_filepath": "/path/to/audio.wav",
  "segments": [
    {"turn": "user", "start_time": 0.0, "end_time": 0.5, "text": "Hi"},
    {"turn": "user-end", "start_time": 0.5, "end_time": 2.5, "text": ""},
    {"turn": "system", "start_time": 2.5, "end_time": 3.2, "text": "Hello"},
    {"turn": "system-end", "start_time": 3.2, "end_time": 4.3, "text": ""},
    {"turn": "user", "start_time": 4.3, "end_time": 8.5, "text": "Hi, I need some help with .."},
    {"turn": "user-end", "start_time": 8.5, "end_time": 9.1, "text": ""},
    {"turn": "system", "start_time": 9.1, "end_time": 12, "text": "Sure, ..."}
  ]
}
```

---

This endpointing is designed for conversational speech between 2 speakers (user and system). The end-of-turn is optimised for user speech only. It supports the following configurations

#### 1. Single-stream with both speakers
Used when both speaker audio is present in single audio stream. We are not interested in perfroming diarisation here, so we provide the timings for the system speech to the model, so that the model can learn to predict turn-ends for user.

#### 2. Single-stream using only user
Used when both speaker audio is present in single audio stream. We mask the loss for the system frames. While the model can learn from interaction with system, the loss is only computed for user and user-end.

#### 3. Dual-stream using only user
Here, we have access to seperate audio streams for user and system. We ignore the system stream, and train the single-stream endpointer only on user speech.

#### 4. Dual-stream using both speakers
Here, we have access to seperate audio streams for user and system. Here, we use a modified architecture with 2 LSTM modules to learn user and system specific features, and learn to predict user, user-end, system, system-end.

NOTE: This could be simplied by treating user and system as the same turn. However, we avoid this because depending on the dataset, user and system have varying pause patterns. System tends to have much larger pauses (They are processing user query, looking up information, etc).

The choice of endpointer would depend on the application. For example, a user-only endpointer could be used for speech recogntion. A user-system endpointer could be used for full-duplex integration (however, with additional overhead for waiting for both user and system frames).

---




For any queries, feel free to reach out to udupa@fit.vutbr.cz / sathvikudupa66@gmail.com

