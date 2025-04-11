import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import librosa
import scipy.signal
import soundfile as sf
import matplotlib.pyplot as plt

# Define constants
LATENT_DIM = 100
SR = 16000  # Sample rate
N_FFT = 1024
HOP_LENGTH = 512
N_MELS = 128
TIME_DIM = 313

# First define the custom layer that was in your original model
class SelfAttention(layers.Layer):
    def __init__(self, **kwargs):
        super(SelfAttention, self).__init__(**kwargs)
        self.gamma = tf.Variable(0.0, trainable=True)
        
    def build(self, input_shape):
        channels = input_shape[-1]
        self.query_conv = layers.Conv2D(channels // 8, 1, padding='same')
        self.key_conv = layers.Conv2D(channels // 8, 1, padding='same')
        self.value_conv = layers.Conv2D(channels, 1, padding='same')
        super(SelfAttention, self).build(input_shape)
        
    def call(self, x):
        batch_size, height, width, channels = tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2], tf.shape(x)[3]
        
        # Apply convolutions to get query, key, value
        query = self.query_conv(x)
        key = self.key_conv(x)
        value = self.value_conv(x)
        
        # Reshape for matrix multiplication
        query_reshaped = tf.reshape(query, [-1, height * width, channels // 8])
        key_reshaped = tf.reshape(key, [-1, height * width, channels // 8])
        value_reshaped = tf.reshape(value, [-1, height * width, channels])
        
        # Calculate attention scores
        scores = tf.matmul(query_reshaped, tf.transpose(key_reshaped, [0, 2, 1]))
        attention_weights = tf.nn.softmax(scores, axis=-1)
        
        # Apply attention
        output = tf.matmul(attention_weights, value_reshaped)
        output = tf.reshape(output, [-1, height, width, channels])
        
        # Add residual connection with learnable parameter
        return x + self.gamma * output
    
    def get_config(self):
        config = super(SelfAttention, self).get_config()
        return config

# Define ExactOutputSize class which was also in your original model
class ExactOutputSize(layers.Layer):
    def __init__(self, target_height, target_width, **kwargs):
        super(ExactOutputSize, self).__init__(**kwargs)
        self.target_height = target_height
        self.target_width = target_width
        
    def build(self, input_shape):
        super(ExactOutputSize, self).build(input_shape)
        
    def call(self, inputs):
        return tf.image.resize(inputs, [self.target_height, self.target_width])
    
    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.target_height, self.target_width, input_shape[3])
        
    def get_config(self):
        config = super(ExactOutputSize, self).get_config()
        config.update({
            'target_height': self.target_height,
            'target_width': self.target_width
        })
        return config

# Define the improved spectrogram to audio conversion function
def improved_spectrogram_to_audio(spectrogram, original_audio_sample=None):
    """
    Convert a generated spectrogram back to audio with focus on preserving 
    the quieter characteristics of fan sounds
    """
    # Remove extra dimensions
    spectrogram = spectrogram.squeeze()
    
    # Convert from normalized range back to appropriate dB range
    # For tanh output in [-1, 1] range
    if np.min(spectrogram) < 0:
        # Use a lower scale factor to reduce amplitude (20 instead of 40)
        db_spectrogram = spectrogram * 20  # Reduced scale for quieter sounds
    else:
        # For sigmoid output [0, 1], map to [-80, 0] but with lower overall volume
        db_spectrogram = spectrogram * 60 - 60  # Adjusted for quieter sounds
    
    # Convert from dB back to power with careful scaling
    S = librosa.db_to_power(db_spectrogram)
    
    # Use Griffin-Lim algorithm with more iterations for better phase reconstruction
    y = librosa.feature.inverse.mel_to_audio(
        S, 
        sr=SR, 
        n_fft=N_FFT, 
        hop_length=HOP_LENGTH,
        power=2.0,
        n_iter=64  # More iterations for better quality
    )
    
    # Apply post-processing tailored for fan sounds
    
    # 1. Apply a gentle low-pass filter to smooth harsh frequencies (fan sounds are usually low frequency)
    def butter_lowpass(cutoff, fs, order=5):
        nyq = 0.5 * fs
        normal_cutoff = cutoff / nyq
        b, a = scipy.signal.butter(order, normal_cutoff, btype='low', analog=False)
        return b, a
    
    def lowpass_filter(data, cutoff, fs, order=5):
        b, a = butter_lowpass(cutoff, fs, order=order)
        y = scipy.signal.filtfilt(b, a, data)
        return y
    
    # Filter frequencies above 2000Hz for fan sounds
    y = lowpass_filter(y, 2000, SR)
    
    # 2. Use original audio for amplitude reference if available
    if original_audio_sample is not None:
        # Match the RMS energy level with original audio
        target_rms = np.sqrt(np.mean(original_audio_sample**2))
        current_rms = np.sqrt(np.mean(y**2))
        gain = target_rms / (current_rms + 1e-8)
        y = y * gain
    else:
        # If no reference, make sure it's quiet (fan-appropriate)
        y = y / (np.max(np.abs(y)) + 1e-8) * 0.4  # Lower scaling factor (0.4 instead of 0.9)
    
    # 3. Add subtle background noise to mask digital artifacts
    # Create quiet pink noise (better for fan-like sounds than white noise)
    def generate_pink_noise(length, amplitude=0.005):
        white_noise = np.random.normal(0, 1, length)
        # Create pink noise by applying 1/f filter
        b = [0.049922035, -0.095993537, 0.050612699, -0.004408786]
        a = [1, -2.494956002, 2.017265875, -0.522189400]
        pink_noise = scipy.signal.lfilter(b, a, white_noise)
        return pink_noise * amplitude
    
    # Add quiet pink noise
    background_noise = generate_pink_noise(len(y))
    y = y + background_noise
    
    # 4. Apply subtle smoothing filter
    y = scipy.signal.savgol_filter(y, 11, 3)
    
    return y

# Fan-specific post-processing for better audio quality
def apply_fan_post_processing(audio, is_abnormal=False, reference_audio=None):
    """Apply fan-specific post-processing to make generated audio sound more realistic"""
    
    # 1. Match amplitude characteristics of reference if available
    if reference_audio is not None:
        target_rms = np.sqrt(np.mean(reference_audio**2))
        current_rms = np.sqrt(np.mean(audio**2))
        gain = target_rms / (current_rms + 1e-8)
        audio = audio * gain
    else:
        # Default to quiet levels typical for fan recordings
        audio = audio / (np.max(np.abs(audio)) + 1e-8) * 0.4
    
    # 2. Apply specific processing based on normal/abnormal condition
    if not is_abnormal:
        # For normal fan sound: smooth, consistent, periodic
        
        # Apply bandpass filter to focus on typical fan frequencies (80-800 Hz)
        def butter_bandpass(lowcut, highcut, fs, order=5):
            nyq = 0.5 * fs
            low = lowcut / nyq
            high = highcut / nyq
            b, a = scipy.signal.butter(order, [low, high], btype='band')
            return b, a
        
        def bandpass_filter(data, lowcut, highcut, fs, order=5):
            b, a = butter_bandpass(lowcut, highcut, fs, order=order)
            y = scipy.signal.filtfilt(b, a, data)
            return y
        
        # Apply bandpass filter focusing on fan frequencies
        audio = bandpass_filter(audio, 80, 800, SR, order=3)
        
        # Add subtle periodicity enhancement for fan sound
        # Try to detect and enhance fan rotation periodicity
        from scipy.signal import find_peaks
        peaks, _ = find_peaks(audio, distance=SR//100)  # Look for peaks at least 10ms apart
        
        if len(peaks) > 5:
            # Calculate median distance between peaks (potential fan rotation period)
            peak_distances = np.diff(peaks)
            median_distance = np.median(peak_distances)
            
            # If we have a reasonable fan frequency (2-20 Hz, typical for fans)
            if SR/median_distance < 20 and SR/median_distance > 2:
                # Add subtle harmonic enhancement at detected frequency
                fan_freq = SR / median_distance
                
                # Create subtle harmonic enhancement at fan frequency
                t = np.arange(len(audio)) / SR
                enhancement = 0.05 * np.sin(2 * np.pi * fan_freq * t)
                
                # Apply enhancement with a gentle fade-in
                fade_in = np.linspace(0, 1, min(int(SR * 0.5), len(audio)))
                if len(fade_in) < len(audio):
                    fade_in = np.pad(fade_in, (0, len(audio) - len(fade_in)), 'constant', constant_values=1)
                
                audio = audio + enhancement * fade_in
    else:
        # For abnormal fan sound: add irregular elements
        
        # Generate random impulses to simulate bearing problems
        impulse_count = np.random.randint(3, 8)
        for _ in range(impulse_count):
            position = np.random.randint(0, len(audio) - SR//4)
            duration = np.random.randint(SR//100, SR//20)
            
            # Create impulse shape (quick attack, slower decay)
            impulse = np.zeros(duration)
            attack = duration // 5
            decay = duration - attack
            impulse[:attack] = np.linspace(0, 1, attack)
            impulse[attack:] = np.linspace(1, 0, decay) ** 2
            
            # Add impulse to audio
            impulse_gain = np.random.uniform(0.1, 0.3)
            end_idx = min(position + duration, len(audio))
            impulse_len = end_idx - position
            audio[position:end_idx] += impulse_gain * impulse[:impulse_len]
        
        # Add some low-frequency rumble for abnormal fan
        t = np.arange(len(audio)) / SR
        rumble_freq = np.random.uniform(20, 50)  # Random low frequency
        rumble = 0.1 * np.sin(2 * np.pi * rumble_freq * t)
        audio += rumble
    
    # 3. Add subtle ambient noise (room tone) typical in fan recordings
    def generate_room_tone(length, amplitude=0.02):
        # Create colored noise for realistic room tone
        white_noise = np.random.normal(0, 1, length)
        # Apply filter to create room-like colored noise
        b, a = scipy.signal.butter(2, 0.1, btype='low')
        room_tone = scipy.signal.filtfilt(b, a, white_noise)
        return room_tone * amplitude
    
    # Add very quiet room tone
    room_tone = generate_room_tone(len(audio))
    audio += room_tone
    
    # 4. Final normalization and limiting
    audio = audio / (np.max(np.abs(audio)) + 1e-8) * 0.95
    
    # 5. Gentle compression
    def apply_compression(audio, threshold=0.5, ratio=4.0):
        # Simple compressor
        compressed = np.zeros_like(audio)
        for i, sample in enumerate(audio):
            if abs(sample) > threshold:
                if sample > 0:
                    compressed[i] = threshold + (sample - threshold) / ratio
                else:
                    compressed[i] = -threshold + (sample + threshold) / ratio
            else:
                compressed[i] = sample
        return compressed
    
    audio = apply_compression(audio)
    
    # 6. Final normalization to match target levels
    if reference_audio is not None:
        target_rms = np.sqrt(np.mean(reference_audio**2))
        current_rms = np.sqrt(np.mean(audio**2))
        audio = audio * (target_rms / (current_rms + 1e-8))
    else:
        # Default quiet levels
        audio = audio / (np.max(np.abs(audio)) + 1e-8) * 0.4
    
    return audio

# Main script
if __name__ == "__main__":
    try:
        # Register the custom layers so Keras can load them properly
        # This is the key step that was missing in your original script
        custom_objects = {
            'SelfAttention': SelfAttention,
            'ExactOutputSize': ExactOutputSize
        }
        
        # Try to load an original fan recording for reference
        try:
            print("Looking for reference fan audio...")
            import glob
            import os
            
            # Try to find fan audio files
            fan_files = glob.glob('**/trimmed_fan/normal/**/**.wav', recursive=True)
            
            if fan_files:
                print(f"Found reference file: {fan_files[0]}")
                reference_audio, _ = librosa.load(fan_files[0], sr=SR)
            else:
                print("No reference audio found. Using default parameters.")
                reference_audio = None
        except Exception as e:
            print(f"Error finding reference audio: {e}")
            reference_audio = None
        
        # Load the model with custom objects
        with tf.keras.utils.custom_object_scope(custom_objects):
            print("Loading generator model...")
            your_generator = tf.keras.models.load_model('fan_generator.h5')
            print("Model loaded successfully!")
        
        # Generate a spectrogram
        print("Generating fan sound...")
        noise = np.random.normal(0, 1, (1, LATENT_DIM))
        
        # Generate both normal and abnormal
        for condition_value, condition_name in enumerate(['normal', 'abnormal']):
            condition = np.array([[condition_value]])  # 0 for normal, 1 for abnormal
            generated_spectrogram = your_generator.predict([noise, condition])
            
            # Convert to audio using improved method
            print(f"Converting {condition_name} spectrogram to audio...")
            audio = improved_spectrogram_to_audio(generated_spectrogram[0], reference_audio)
            
            # Save raw audio
            raw_filename = f'improved_{condition_name}_fan_raw.wav'
            sf.write(raw_filename, audio, SR)
            print(f"Raw audio saved as {raw_filename}")
            
            # Apply fan-specific post-processing
            print(f"Applying post-processing for {condition_name} fan sound...")
            processed_audio = apply_fan_post_processing(
                audio, 
                is_abnormal=(condition_value == 1), 
                reference_audio=reference_audio
            )
            
            # Save processed audio
            processed_filename = f'improved_{condition_name}_fan.wav'
            sf.write(processed_filename, processed_audio, SR)
            print(f"Processed audio saved as {processed_filename}")
            
            # Save spectrogram visualization
            plt.figure(figsize=(10, 6))
            plt.imshow(generated_spectrogram[0, :, :, 0], aspect='auto', origin='lower', cmap='viridis')
            plt.title(f"{condition_name.capitalize()} Fan Sound Spectrogram")
            plt.colorbar(format='%.2f')
            plt.tight_layout()
            plt.savefig(f'{condition_name}_spectrogram.png', dpi=300)
            plt.close()
            
        print("Process completed successfully!")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()