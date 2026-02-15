# Configuration

### Copying the configuration

{% hint style="info" %}
#### Navigate to the Correct Folder

Ensure you are in the `VoiceSentinel/processor-voicesentinel` directory before proceeding.
{% endhint %}

To copy `config.example.json` to `config.json`, run the following command:

```
cp config.example.json config.json
```

**Configuration Example (Pre-installed)**

```json
{
  "server": {
    "host": "0.0.0.0",
    "port": 8000,
    "workers": 4,
    "cors": {
      "allow_origins": ["<your-transcriber-url>"],
      "allow_credentials": true,
      "allow_methods": ["GET", "POST", "DELETE"],
      "allow_headers": [
        "Content-Type", 
        "Authorization", 
        "Accept",
        "User-Agent",
        "X-Requested-With"
      ]
    }
  },
  "security": {
    "api_keys": [],
    "rate_limit": {
      "authenticated": {
        "window_seconds": 60,
        "max_requests": 1000,
        "block_duration_seconds": 300
      },
      "unauthenticated": {
        "window_seconds": 60,
        "max_requests": 10,
        "block_duration_seconds": 1800
      }
    }
  },
  "transcription": {
    "engine": "whisper",
    "model": "tiny.en",
    "language": "en",
    "timeout_seconds": 90,
    "cpu_threads": 1,
    "options": {
      "fp16": false,
      "temperature": 0.0,
      "condition_on_previous_text": false,
      "no_speech_threshold": 0.7,
      "beam_size": 1,
      "best_of": 1,
      "suppress_blank": true,
      "initial_prompt": "Player voice chat audio containing speech."
    }
  },
  "processing": {
    "max_concurrent_jobs": 4, 
    "queue_warning_threshold": 200,
    "max_queue_size": 1000,
    "timeout_seconds": 90,
    "retry_attempts": 2,
    "retry_delay_seconds": 1
  },
  "audio": {
    "max_chunk_size": 262144,
    "max_total_size": 5242880,
    "sample_rate": 16000,
    "channels": 1,
    "min_length_ms": 300,
    "max_length_ms": 15000
  }
} 
```

The configuration is pretty self explanatory, what you should pay attention to:

* `max_concurrent_jobs`: Sets the maximum number of jobs that can run at the same time.
* `max_queue_size`: Determines the largest number of jobs that can be queued for processing.
* `cpu_threads`: Configures the number of CPU threads available for processing tasks. You shouldn't need much, it depends on your playerbase.
* `cors`: Enables Cross-Origin Resource Sharing (CORS) settings. See [Securing your processor](securing-your-processor.md).
* `workers`: Specifies the number of worker processes handling tasks. Increase threads = Add workers.

### Setting your server key(s)

{% hint style="info" %}
For generating keys, I recommend using [jwtsecrets.com](https://jwtsecrets.com/).
{% endhint %}

#### Editing the config.json (processor

Place your server key into the array like this:

```json
  "security": {
    "api_keys": ["your-generated-key-here"],
```

#### Editing the config.yml (plugin)

```yaml
# Unique server key for authentication (must be 16+ characters)
# SECURITY: Change this to a strong, unique value for your server!
# Use a password generator to create a secure key
server_key: "your-generated-key-here"
```
