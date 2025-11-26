# NAC-LD-Endpointer
Codebase for the work "Streaming Endpointer for Spoken Dialogue using Neural Audio Codecs and Label-Delayed Training", accepted at ASRU 2025

# TODO
- [ ] Add training code
- [ ] Add inference script
- [ ] Add Moshi integration script

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

**Example processed segment:**
```json
{
  "audio_filepath": "/path/to/audio.wav",
  "segments": [
    {"turn": "user", "start_time": 0.0, "end_time": 0.5, "text": "Hi"},
    {"turn": "system", "start_time": 2.5, "end_time": 3.2, "text": "Hello"},
    {"turn": "user", "start_time": 4.3, "end_time": 8.5, "text": "Hi, I need some help with .."},
    {"turn": "system", "start_time": 9.1, "end_time": 12, "text": "Sure, ..."}
  ]
}
```

For any queries, feel free to reach out to udupa@fit.vutbr.cz / sathvikudupa66@gmail.com

