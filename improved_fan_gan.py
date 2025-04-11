# 1. Enhance the spectrogram_to_audio function for smoother conversion
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
        y = y * (target_rms / (current_rms + 1e-8))
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

# 2. Create a better load_and_preprocess_audio function specifically for fan sounds
def load_and_preprocess_fan_audio(file_path, is_abnormal=False, reference_audio=None):
    """
    Load audio file and convert to mel spectrogram - optimized for fan sounds
    """
    # Load audio file
    audio, _ = librosa.load(file_path, sr=SR, duration=AUDIO_LENGTH)
    
    # Store reference audio if needed
    if reference_audio is None and not is_abnormal:
        reference_audio = audio.copy()
    
    # Normalize audio more gently to preserve quiet characteristics
    audio = audio / (np.max(np.abs(audio)) + 1e-8)
    
    # For fan sounds, apply a subtle low-pass filter to focus on fan frequencies
    def butter_lowpass(cutoff, fs, order=5):
        nyq = 0.5 * fs
        normal_cutoff = cutoff / nyq
        b, a = scipy.signal.butter(order, normal_cutoff, btype='low', analog=False)
        return b, a
    
    def lowpass_filter(data, cutoff, fs, order=5):
        b, a = butter_lowpass(cutoff, fs, order=order)
        y = scipy.signal.filtfilt(b, a, data)
        return y
    
    # Apply fan-specific pre-emphasis (emphasize mid-range frequencies)
    audio = lowpass_filter(audio, 2000, SR)
    
    # Pad if needed
    if len(audio) < SR * AUDIO_LENGTH:
        audio = np.pad(audio, (0, SR * AUDIO_LENGTH - len(audio)))
    elif len(audio) > SR * AUDIO_LENGTH:
        audio = audio[:SR * AUDIO_LENGTH]
    
    # Create mel spectrogram with fan-optimized parameters
    mel_spec = librosa.feature.melspectrogram(
        y=audio, 
        sr=SR, 
        n_fft=N_FFT, 
        hop_length=HOP_LENGTH, 
        n_mels=N_MELS,
        fmin=20,  # Keep low frequency range for fan sounds
        fmax=4000  # Limit upper range since fan sounds are mostly low frequency
    )
    
    # Convert to log scale
    log_mel_spec = librosa.power_to_db(mel_spec)
    
    # Normalize to [-1, 1] range
    log_mel_spec_min = log_mel_spec.min()
    log_mel_spec_max = log_mel_spec.max()
    log_mel_spec_norm = 2 * (log_mel_spec - log_mel_spec_min) / (log_mel_spec_max - log_mel_spec_min) - 1
    
    # Add channel dimension
    log_mel_spec_norm = np.expand_dims(log_mel_spec_norm, axis=-1)
    
    # Pad to match expected shape if needed
    if log_mel_spec_norm.shape[1] < TIME_DIM:
        pad_width = ((0, 0), (0, TIME_DIM - log_mel_spec_norm.shape[1]), (0, 0))
        log_mel_spec_norm = np.pad(log_mel_spec_norm, pad_width, mode='constant')
    elif log_mel_spec_norm.shape[1] > TIME_DIM:
        log_mel_spec_norm = log_mel_spec_norm[:, :TIME_DIM, :]
    
    # Add label for condition
    condition = 1 if is_abnormal else 0
    
    return log_mel_spec_norm, condition, reference_audio

# 3. Enhanced generator architecture with fan-specific modifications
def build_fan_generator(latent_dim):
    """Build improved generator network specifically tuned for fan sounds"""
    # Input for latent space
    noise = layers.Input(shape=(latent_dim,), name="noise_input")
    
    # Input for conditions (normal/abnormal)
    condition = layers.Input(shape=(1,), name="condition_input")
    
    # Embed the condition with higher dimension
    condition_embedding = layers.Embedding(2, 64)(condition)
    condition_embedding = layers.Flatten()(condition_embedding)
    
    # Concatenate noise and condition
    combined_input = layers.Concatenate()([noise, condition_embedding])
    
    # Dense layers with normalization and better activation
    x = layers.Dense(256)(combined_input)
    x = layers.BatchNormalization(momentum=0.8)(x)
    x = layers.LeakyReLU(alpha=0.2)(x)
    
    x = layers.Dense(512)(x)
    x = layers.BatchNormalization(momentum=0.8)(x)
    x = layers.LeakyReLU(alpha=0.2)(x)
    
    # Calculate dimensions for reshaping
    target_height = N_MELS // 8  
    target_width = TIME_DIM // 16
    
    x = layers.Dense(target_height * target_width * 64)(x)
    x = layers.BatchNormalization(momentum=0.8)(x)
    x = layers.LeakyReLU(alpha=0.2)(x)
    x = layers.Reshape((target_height, target_width, 64))(x)
    
    # First transposed convolution with residual path
    residual = x
    x = layers.Conv2DTranspose(64, (4, 4), strides=(2, 2), padding='same')(x)
    x = layers.BatchNormalization(momentum=0.8)(x)
    x = layers.LeakyReLU(alpha=0.2)(x)
    
    # Second transposed convolution
    x = layers.Conv2DTranspose(32, (4, 4), strides=(2, 2), padding='same')(x)
    x = layers.BatchNormalization(momentum=0.8)(x)
    x = layers.LeakyReLU(alpha=0.2)(x)
    # Add self-attention after this layer
    x = SelfAttention()(x)
    
    # Third transposed convolution
    x = layers.Conv2DTranspose(16, (4, 4), strides=(2, 2), padding='same')(x)
    x = layers.BatchNormalization(momentum=0.8)(x)
    x = layers.LeakyReLU(alpha=0.2)(x)
    
    # Fourth transposed convolution
    x = layers.Conv2DTranspose(8, (4, 4), strides=(2, 2), padding='same')(x)
    x = layers.BatchNormalization(momentum=0.8)(x)
    x = layers.LeakyReLU(alpha=0.2)(x)
    
    # Add another residual connection for better feature preservation
    # This helps maintain the subtle features of quiet fan sounds
    skip_connection = layers.Conv2DTranspose(8, (8, 8), strides=(8, 8), padding='same')(residual)
    x = layers.Add()([x, skip_connection])
    
    # Ensure exact output dimensions
    x = ExactOutputSize(N_MELS, TIME_DIM)(x)
    
    # Use tanh activation for better capturing subtle variations
    output = layers.Conv2D(1, (3, 3), activation='tanh', padding='same')(x)
    
    # Model
    model = models.Model([noise, condition], output, name="fan_generator")
    
    return model

# 4. Improved discriminator with frequency attention for fan-specific features
class FrequencyAttention(layers.Layer):
    """Custom attention layer that focuses on important frequency bands for fan sounds"""
    def __init__(self, **kwargs):
        super(FrequencyAttention, self).__init__(**kwargs)
        
    def build(self, input_shape):
        self.gamma = self.add_weight(name='gamma', 
                                      shape=(1,),
                                      initializer='zeros',
                                      trainable=True)
        self.dense_1 = layers.Dense(input_shape[1], activation='relu')
        self.dense_2 = layers.Dense(input_shape[1], activation='sigmoid')
        super(FrequencyAttention, self).build(input_shape)
        
    def call(self, x):
        # Compute frequency attention weights by averaging across time
        freq_features = tf.reduce_mean(x, axis=2)  # Average across time
        attention = self.dense_1(freq_features)
        attention = self.dense_2(attention)
        
        # Reshape for broadcasting
        attention = tf.expand_dims(attention, axis=2)
        attention = tf.repeat(attention, tf.shape(x)[2], axis=2)
        
        # Apply attention
        return x + self.gamma * x * attention

def build_fan_discriminator():
    """Build improved discriminator network optimized for fan sounds"""
    # Input for spectrogram
    input_spec = layers.Input(shape=(N_MELS, TIME_DIM, 1), name="spec_input")
    
    # Input for conditions
    condition = layers.Input(shape=(1,), name="condition_input")
    
    # Embedding layer for condition
    embedding_dim = N_MELS * TIME_DIM
    condition_embedding = layers.Embedding(2, embedding_dim)(condition)
    condition_embedding = layers.Reshape((N_MELS, TIME_DIM, 1))(condition_embedding)
    
    # Concatenate inputs
    combined_input = layers.Concatenate(axis=-1)([input_spec, condition_embedding])
    
    # Initial layers
    x = layers.Conv2D(32, (5, 5), strides=(2, 2), padding='same')(combined_input)
    x = layers.LeakyReLU(alpha=0.2)(x)
    x = layers.Dropout(0.3)(x)
    
    # Add frequency attention to focus on fan-specific frequency bands
    x = FrequencyAttention()(x)
    
    x = layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same')(x)
    x = layers.BatchNormalization(momentum=0.8)(x)
    x = layers.LeakyReLU(alpha=0.2)(x)
    x = layers.Dropout(0.3)(x)
    
    x = layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same')(x)
    x = layers.BatchNormalization(momentum=0.8)(x)
    x = layers.LeakyReLU(alpha=0.2)(x)
    x = layers.Dropout(0.3)(x)
    
    x = layers.Conv2D(256, (5, 5), strides=(2, 2), padding='same')(x)
    x = layers.BatchNormalization(momentum=0.8)(x)
    x = layers.LeakyReLU(alpha=0.2)(x)
    x = layers.Dropout(0.3)(x)
    
    # Add standard self-attention
    x = SelfAttention()(x)
    
    # Flatten and dense layers
    x = layers.Flatten()(x)
    x = layers.Dense(512)(x)
    x = layers.LeakyReLU(alpha=0.2)(x)
    x = layers.Dropout(0.3)(x)
    
    # Output layer
    output = layers.Dense(1, activation='sigmoid')(x)
    
    # Model
    model = models.Model([input_spec, condition], output, name="fan_discriminator")
    
    return model

# 5. Improved GAN training function with frequency-focused loss for fan sounds
def train_fan_gan(generator, discriminator, gan, specs, conditions, epochs, batch_size, reference_audio=None):
    """Train the GAN model with special focus on preserving fan sound characteristics"""
    
    # Create folder for samples
    output_dir = "generated_samples_fan"
    os.makedirs(output_dir, exist_ok=True)
    
    # Arrays to store loss values
    d_losses = []
    g_losses = []
    
    # Create EMA model for better stability
    ema_generator = tf.keras.models.clone_model(generator)
    ema_generator.set_weights(generator.get_weights())
    
    # Create spectral loss function for better frequency matching
    def spectral_convergence_loss(y_true, y_pred):
        """Computes spectral convergence loss between original and generated spectrograms"""
        # Remove channel dimension
        y_true = tf.squeeze(y_true, axis=-1)
        y_pred = tf.squeeze(y_pred, axis=-1)
        
        # Calculate magnitude difference
        magnitudes_true = tf.abs(y_true)
        magnitudes_pred = tf.abs(y_pred)
        
        # Calculate Frobenius norm of difference
        diff_norm = tf.norm(magnitudes_true - magnitudes_pred, ord='fro', axis=(1,2))
        true_norm = tf.norm(magnitudes_true, ord='fro', axis=(1,2))
        
        # Return spectral convergence
        return diff_norm / (true_norm + 1e-10)
    
    # Training loop with improved techniques for fan sounds
    for epoch in range(epochs):
        # -----------------
        # Train Discriminator
        # -----------------
        
        # Select a balanced batch of real spectrograms
        normal_indices = np.where(conditions == 0)[0]
        abnormal_indices = np.where(conditions == 1)[0]
        
        normal_count = min(batch_size // 2, len(normal_indices))
        abnormal_count = min(batch_size // 2, len(abnormal_indices))
        
        selected_normal = np.random.choice(normal_indices, normal_count, replace=False)
        selected_abnormal = np.random.choice(abnormal_indices, abnormal_count, replace=False)
        
        idx = np.concatenate([selected_normal, selected_abnormal])
        np.random.shuffle(idx)
        
        real_specs = specs[idx]
        real_conditions = conditions[idx].reshape(-1, 1)
        
        # Generate batch of fake spectrograms
        noise = np.random.normal(0, 1, (len(idx), LATENT_DIM))
        fake_conditions = real_conditions.copy()
        
        # Generate fake spectrograms
        fake_specs = generator.predict([noise, fake_conditions], verbose=0)
        
        # Use one-sided label smoothing for more stable training
        real_labels = np.ones((len(idx), 1)) * 0.9  # 0.9 instead of 1.0
        fake_labels = np.zeros((len(idx), 1))
        
        # For quiet fan sounds, add some noise to labels to prevent mode collapse
        real_labels += np.random.normal(0, 0.05, real_labels.shape)
        fake_labels += np.random.normal(0, 0.05, fake_labels.shape)
        
        # Clip to valid range
        real_labels = np.clip(real_labels, 0, 1)
        fake_labels = np.clip(fake_labels, 0, 1)
        
        # Train discriminator
        d_loss_real = discriminator.train_on_batch([real_specs, real_conditions], real_labels)
        d_loss_fake = discriminator.train_on_batch([fake_specs, fake_conditions], fake_labels)
        
        # Calculate discriminator loss
        if isinstance(d_loss_real, list):
            d_loss = 0.5 * (d_loss_real[0] + d_loss_fake[0])
        else:
            d_loss = 0.5 * (d_loss_real + d_loss_fake)
        
        # -----------------
        # Train Generator with fan-specific techniques
        # -----------------
        
        # Generate new batch of noise and conditions
        noise = np.random.normal(0, 1, (batch_size, LATENT_DIM))
        
        # Create balanced conditions
        random_conditions = np.zeros((batch_size, 1))
        random_conditions[batch_size//2:] = 1
        np.random.shuffle(random_conditions)
        
        # Train generator
        g_loss = gan.train_on_batch([noise, random_conditions], np.ones((batch_size, 1)))
        
        # Update EMA generator weights
        decay = 0.999
        generator_weights = generator.get_weights()
        ema_weights = ema_generator.get_weights()
        
        for i in range(len(generator_weights)):
            ema_weights[i] = decay * ema_weights[i] + (1.0 - decay) * generator_weights[i]
            
        ema_generator.set_weights(ema_weights)
        
        # Store losses
        d_losses.append(d_loss)
        g_losses.append(g_loss)
        
        # Print progress
        if isinstance(d_loss, np.ndarray):
            d_loss_val = float(d_loss)
        else:
            d_loss_val = d_loss
            
        if isinstance(g_loss, np.ndarray):
            g_loss_val = float(g_loss)
        else:
            g_loss_val = g_loss
            
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"[Fan] Epoch {epoch+1}/{epochs} | D Loss: {d_loss_val:.4f} | G Loss: {g_loss_val:.4f}")
        
        # Save samples at specific milestones
        if (epoch + 1) in [1, 100, 500, 1000, 2000] or (epoch + 1) == epochs:
            print(f"\nGenerating samples for epoch {epoch+1}...")
            save_fan_samples(ema_generator, epoch + 1, output_dir, reference_audio)
    
    # Save final model
    generator.save("fan_generator.h5")
    discriminator.save("fan_discriminator.h5")
    ema_generator.save("fan_ema_generator.h5")
    
    # Generate final samples
    generate_final_fan_samples(ema_generator, output_dir, reference_audio)
    
    return d_losses, g_losses

# 6. Improved sample generation function for fan sounds
def save_fan_samples(generator, epoch, output_dir, reference_audio=None):
    """Save generated fan sound spectrograms and audio samples with improved quality"""
    # Generate noise and conditions
    noise = np.random.normal(0, 1, (2, LATENT_DIM))
    conditions = np.array([[0], [1]])  # Normal and abnormal
    
    # Generate spectrograms
    gen_specs = generator.predict([noise, conditions], verbose=0)
    
    # Save combined spectrograms as images
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.imshow(gen_specs[0, :, :, 0], aspect='auto', origin='lower', cmap='viridis')
    plt.title("Normal Fan Sound")
    plt.colorbar(format='%.2f')
    plt.xlabel('Time Frame')
    plt.ylabel('Mel Frequency Bin')
    
    plt.subplot(1, 2, 2)
    plt.imshow(gen_specs[1, :, :, 0], aspect='auto', origin='lower', cmap='viridis')
    plt.title("Abnormal Fan Sound")
    plt.colorbar(format='%.2f')
    plt.xlabel('Time Frame')
    plt.ylabel('Mel Frequency Bin')
    
    plt.tight_layout()
    
    plt.savefig(f"{comparison_dir}/fan_sound_comparison.png", dpi=300)
    plt.close()
    
    # Save audio files
    sf.write(f"{comparison_dir}/original_fan.wav", reference_stats['audio'], SR)
    sf.write(f"{comparison_dir}/generated_fan.wav", gen_audio, SR)
    
    print(f"Comparison analysis saved to {comparison_dir}/fan_sound_comparison.png")
    print(f"Original and generated audio saved to {comparison_dir}")
    
    # Print quantitative analysis of the match
    rms_ratio = gen_rms / reference_stats['rms']
    centroid_ratio = np.mean(gen_centroid) / reference_stats['centroid']
    
    print("\n=== Fan Sound Generation Quality Analysis ===")
    print(f"RMS Amplitude Match: {min(rms_ratio, 1/rms_ratio):.2%} similarity")
    print(f"Frequency Centroid Match: {min(centroid_ratio, 1/centroid_ratio):.2%} similarity")
    
    # Create a hybrid audio combining original and generated sounds
    # This can help demonstrate the differences while still maintaining fan character
    hybrid_audio = np.zeros_like(reference_stats['audio'])
    fade = np.linspace(0, 1, len(hybrid_audio))
    hybrid_audio = fade * gen_audio[:len(hybrid_audio)] + (1 - fade) * reference_stats['audio']
    
    sf.write(f"{comparison_dir}/hybrid_transition_fan.wav", hybrid_audio, SR)
    print(f"Transition demonstration saved to {comparison_dir}/hybrid_transition_fan.wav")

# 11. Implementation of Wasserstein GAN with Gradient Penalty (WGAN-GP) for fan sounds
class WGANFanTrainer:
    """WGAN-GP trainer specifically optimized for fan sound generation"""
    
    def __init__(self, generator, discriminator, latent_dim, sr=16000):
        self.generator = generator
        self.discriminator = discriminator
        self.latent_dim = latent_dim
        self.sr = sr
        self.n_critic = 5  # Number of critic updates per generator update
        self.gp_weight = 10.0  # Weight for gradient penalty
        
        # Replace sigmoid with linear activation in discriminator
        self.modify_discriminator_for_wgan()
        
        # Optimizers
        self.g_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001, beta_1=0, beta_2=0.9)
        self.d_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001, beta_1=0, beta_2=0.9)
        
        # Create EMA model
        self.ema_generator = tf.keras.models.clone_model(generator)
        self.ema_generator.set_weights(generator.get_weights())
        self.ema_decay = 0.999
    
    def modify_discriminator_for_wgan(self):
        """Modify the discriminator to work with Wasserstein loss"""
        # Get the layers without the final activation
        layers = [layer for layer in self.discriminator.layers]
        
        # Find the output layer and replace it
        for i, layer in enumerate(layers):
            if isinstance(layer, tf.keras.layers.Dense) and layer.name == self.discriminator.output.name:
                # Replace with linear activation
                config = layer.get_config()
                config['activation'] = 'linear'  # Change to linear activation
                new_layer = tf.keras.layers.Dense.from_config(config)
                new_layer.build(layer.input_shape)
                new_layer.set_weights(layer.get_weights())
                layers[i] = new_layer
                break
        
        # Rebuild discriminator with modified output
        inputs = self.discriminator.inputs
        x = inputs
        for layer in layers[1:]:  # Skip the input layer
            if isinstance(layer, tf.keras.layers.InputLayer):
                continue
            if hasattr(layer, 'inbound_nodes'):
                if layer.inbound_nodes:
                    inbound_layers = []
                    for node in layer.inbound_nodes:
                        if hasattr(node, 'inbound_layers'):
                            inbound_layers.extend(node.inbound_layers)
                    if all(l in layers[:i] for l in inbound_layers):
                        x = layer(x)
            else:
                x = layer(x)
        
        # Create new model
        self.discriminator = tf.keras.models.Model(inputs, x, name="wgan_critic")
    
    def gradient_penalty(self, real_samples, fake_samples, conditions):
        """Calculate gradient penalty for WGAN-GP"""
        batch_size = tf.shape(real_samples)[0]
        alpha = tf.random.uniform([batch_size, 1, 1, 1], 0.0, 1.0)
        
        # Create interpolated samples
        interpolated = alpha * real_samples + (1 - alpha) * fake_samples
        
        with tf.GradientTape() as gp_tape:
            gp_tape.watch(interpolated)
            # Get critic output for interpolated samples
            critic_interpolated = self.discriminator([interpolated, conditions])
        
        # Calculate gradients w.r.t. interpolated samples
        grads = gp_tape.gradient(critic_interpolated, interpolated)
        # Calculate norm of gradients
        norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2, 3]))
        # Calculate gradient penalty
        gradient_penalty = tf.reduce_mean(tf.square(norm - 1.0))
        
        return gradient_penalty
    
    def train_step(self, real_specs, real_conditions):
        """Perform a single WGAN-GP training step"""
        batch_size = real_specs.shape[0]
        
        # Train discriminator/critic
        for _ in range(self.n_critic):
            with tf.GradientTape() as d_tape:
                # Generate fake samples
                noise = tf.random.normal([batch_size, self.latent_dim])
                fake_specs = self.generator([noise, real_conditions], training=True)
                
                # Get critic scores
                real_output = self.discriminator([real_specs, real_conditions], training=True)
                fake_output = self.discriminator([fake_specs, real_conditions], training=True)
                
                # Calculate Wasserstein loss
                d_loss = tf.reduce_mean(fake_output) - tf.reduce_mean(real_output)
                
                # Calculate gradient penalty
                gp = self.gradient_penalty(real_specs, fake_specs, real_conditions)
                
                # Add gradient penalty to loss
                d_loss = d_loss + self.gp_weight * gp
            
            # Update critic
            d_gradients = d_tape.gradient(d_loss, self.discriminator.trainable_variables)
            self.d_optimizer.apply_gradients(zip(d_gradients, self.discriminator.trainable_variables))
        
        # Train generator
        with tf.GradientTape() as g_tape:
            # Generate fake samples
            noise = tf.random.normal([batch_size, self.latent_dim])
            fake_specs = self.generator([noise, real_conditions], training=True)
            
            # Get critic score
            fake_output = self.discriminator([fake_specs, real_conditions], training=True)
            
            # Calculate generator loss (negative of critic score for fake samples)
            g_loss = -tf.reduce_mean(fake_output)
        
        # Update generator
        g_gradients = g_tape.gradient(g_loss, self.generator.trainable_variables)
        self.g_optimizer.apply_gradients(zip(g_gradients, self.generator.trainable_variables))
        
        # Update EMA generator
        self.update_ema()
        
        return {
            'd_loss': float(d_loss),
            'g_loss': float(g_loss),
        }
    
    def update_ema(self):
        """Update EMA generator weights"""
        generator_weights = self.generator.get_weights()
        ema_weights = self.ema_generator.get_weights()
        
        for i in range(len(generator_weights)):
            ema_weights[i] = self.ema_decay * ema_weights[i] + (1.0 - self.ema_decay) * generator_weights[i]
            
        self.ema_generator.set_weights(ema_weights)
    
    def train(self, specs, conditions, epochs, batch_size, output_dir='wgan_fan_output'):
        """Run WGAN-GP training"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Arrays to store loss values
        history = {
            'd_loss': [],
            'g_loss': []
        }
        
        total_batches = len(specs) // batch_size
        if total_batches == 0:
            total_batches = 1
            
        for epoch in range(epochs):
            # Shuffle data
            indices = np.arange(len(specs))
            np.random.shuffle(indices)
            shuffled_specs = specs[indices]
            shuffled_conditions = conditions[indices].reshape(-1, 1)
            
            # Train on batches
            epoch_d_loss = 0
            epoch_g_loss = 0
            
            for batch in range(total_batches):
                start_idx = batch * batch_size
                end_idx = min((batch + 1) * batch_size, len(specs))
                
                batch_specs = shuffled_specs[start_idx:end_idx]
                batch_conditions = shuffled_conditions[start_idx:end_idx]
                
                # Perform training step
                step_losses = self.train_step(batch_specs, batch_conditions)
                
                # Accumulate losses
                epoch_d_loss += step_losses['d_loss']
                epoch_g_loss += step_losses['g_loss']
            
            # Calculate average losses
            epoch_d_loss /= total_batches
            epoch_g_loss /= total_batches
            
            # Store losses
            history['d_loss'].append(epoch_d_loss)
            history['g_loss'].append(epoch_g_loss)
            
            # Print progress
            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"[WGAN Fan] Epoch {epoch+1}/{epochs} | Critic Loss: {epoch_d_loss:.4f} | "
                      f"Generator Loss: {epoch_g_loss:.4f}")
            
            # Save samples at specific milestones
            if (epoch + 1) in [1, 100, 500, 1000, 2000] or (epoch + 1) == epochs:
                print(f"\nGenerating samples for epoch {epoch+1}...")
                self.save_samples(epoch + 1, output_dir)
        
        # Save final models
        self.generator.save(f"{output_dir}/wgan_fan_generator.h5")
        self.discriminator.save(f"{output_dir}/wgan_fan_critic.h5")
        self.ema_generator.save(f"{output_dir}/wgan_fan_ema_generator.h5")
        
        # Plot loss curves
        plt.figure(figsize=(10, 6))
        plt.plot(history['d_loss'], label='Critic')
        plt.plot(history['g_loss'], label='Generator')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('WGAN-GP Fan Training Loss')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.savefig(f"{output_dir}/wgan_training_loss.png", dpi=300)
        plt.close()
        
        return history
    
    def save_samples(self, epoch, output_dir):
        """Save generated samples using EMA generator"""
        # Generate noise and conditions
        noise = np.random.normal(0, 1, (2, self.latent_dim))
        conditions = np.array([[0], [1]])  # Normal and abnormal
        
        # Generate spectrograms
        gen_specs = self.ema_generator.predict([noise, conditions], verbose=0)
        
        # Save spectrograms
        plt.figure(figsize=(12, 6))
        
        plt.subplot(1, 2, 1)
        plt.imshow(gen_specs[0, :, :, 0], aspect='auto', origin='lower', cmap='viridis')
        plt.title("Normal Fan Sound (WGAN)")
        plt.colorbar(format='%.2f')
        plt.xlabel('Time Frame')
        plt.ylabel('Mel Frequency Bin')
        
        plt.subplot(1, 2, 2)
        plt.imshow(gen_specs[1, :, :, 0], aspect='auto', origin='lower', cmap='viridis')
        plt.title("Abnormal Fan Sound (WGAN)")
        plt.colorbar(format='%.2f')
        plt.xlabel('Time Frame')
        plt.ylabel('Mel Frequency Bin')
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/wgan_spectrograms_epoch_{epoch}.png", dpi=300)
        plt.close()
        
        # Convert to audio with improved function
        for i, condition in enumerate(["normal", "abnormal"]):
            # Convert to audio
            audio = improved_spectrogram_to_audio(gen_specs[i])
            sf.write(f"{output_dir}/wgan_{condition}_epoch_{epoch}.wav", audio, self.sr)

# 12. Fan-specific post-processing function for final audio
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

# 13. Main function to run the complete pipeline
def main(data_dir, method="standard", epochs=1000, batch_size=8):
    """Main function to run the improved fan sound generation pipeline
    
    Args:
        data_dir: Directory containing the MIMII dataset
        method: Training method ("standard", "perceptual", or "wgan")
        epochs: Number of training epochs
        batch_size: Batch size for training
    """
    print("\n==== Fan Sound Generation Pipeline ====")
    print(f"Method: {method}")
    print(f"Data directory: {data_dir}")
    print(f"Epochs: {epochs}")
    print(f"Batch size: {batch_size}")
    print("======================================\n")
    
    # Step 1: Process the fan data
    specs, conditions, reference_samples = process_fan_data(data_dir)
    
    if len(specs) == 0:
        print("Error: No data found. Exiting.")
        return
    
    # Reference audio for amplitude matching
    reference_audio = None
    if reference_samples:
        reference_audio = reference_samples[0]
    
    # Step 2: Build models
    generator = build_fan_generator(LATENT_DIM)
    discriminator = build_fan_discriminator()
    
    # Step 3: Train with selected method
    output_dir = f"fan_sound_gen_{method}"
    os.makedirs(output_dir, exist_ok=True)
    
    if method == "standard":
        # Standard GAN training
        gan = build_gan(generator, discriminator)
        
        discriminator.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001, beta_1=0.5),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        gan.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001, beta_1=0.5),
            loss='binary_crossentropy'
        )
        
        history = train_fan_gan(
            generator, discriminator, gan, specs, conditions, 
            epochs, batch_size, reference_audio
        )
        
    elif method == "perceptual":
        # Perceptual loss training
        trainer = FanGANTrainer(generator, discriminator, LATENT_DIM)
        history = trainer.train(specs, conditions, epochs, batch_size, reference_audio, output_dir)
        generator = trainer.ema_generator  # Use EMA generator for generation
        
    elif method == "wgan":
        # WGAN-GP training
        trainer = WGANFanTrainer(generator, discriminator, LATENT_DIM)
        history = trainer.train(specs, conditions, epochs, batch_size, output_dir)
        generator = trainer.ema_generator  # Use EMA generator for generation
        
    else:
        print(f"Unknown method: {method}")
        return
    
    # Step 4: Generate final samples with full post-processing
    generate_final_fan_samples_with_processing(generator, output_dir, reference_audio)
    
    print("\n==== Fan Sound Generation Complete ====")
    print(f"Final samples saved in {output_dir}/final_samples")
    print("=======================================")

# Helper function to generate final samples with full post-processing
def generate_final_fan_samples_with_processing(generator, output_dir, reference_audio, num_samples=10):
    """Generate final fan sound samples with complete post-processing pipeline"""
    
    final_dir = os.path.join(output_dir, "final_samples")
    os.makedirs(final_dir, exist_ok=True)
    
    for condition_label, condition_name in enumerate(["normal", "abnormal"]):
        condition_dir = os.path.join(final_dir, condition_name)
        os.makedirs(condition_dir, exist_ok=True)
        
        print(f"Generating {num_samples} {condition_name} fan samples with full processing...")
        for i in range(num_samples):
            # Generate with varied latent space
            noise = np.random.normal(0, 1, (1, LATENT_DIM))
            condition = np.array([[condition_label]])
            
            # Generate spectrogram
            gen_spec = generator.predict([noise, condition], verbose=0)
            
            # Convert to audio with improved function
            audio = improved_spectrogram_to_audio(gen_spec[0], reference_audio)
            
            # Apply comprehensive fan-specific post-processing
            processed_audio = apply_fan_post_processing(
                audio, 
                is_abnormal=(condition_label == 1),
                reference_audio=reference_audio
            )
            
            # Save both the raw and processed audio for comparison
            sf.write(f"{condition_dir}/{condition_name}_raw_{i+1}.wav", audio, SR)
            sf.write(f"{condition_dir}/{condition_name}_processed_{i+1}.wav", processed_audio, SR)
            
            # Save spectrogram visualization
            plt.figure(figsize=(10, 6))
            plt.imshow(gen_spec[0, :, :, 0], aspect='auto', origin='lower', cmap='viridis')
            plt.title(f"{condition_name.capitalize()} Fan Sound (Sample {i+1})")
            plt.colorbar(format='%.2f')
            plt.tight_layout()
            plt.savefig(f"{condition_dir}/{condition_name}_spectrogram_{i+1}.png", dpi=300)
            plt.close()
    
    print(f"Fully processed fan samples saved in '{final_dir}'")

# If executed as script
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train improved fan sound generation model')
    parser.add_argument('--data_dir', type=str, required=True, help='Directory containing the MIMII dataset')
    parser.add_argument('--method', type=str, default='perceptual', 
                        choices=['standard', 'perceptual', 'wgan'], 
                        help='GAN training method')
    parser.add_argument('--epochs', type=int, default=1000, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for training')
    
    args = parser.parse_args()
    
    main(args.data_dir, args.method, args.epochs, args.batch_size)
    plt.savefig(f"{output_dir}/spectrograms_epoch_{epoch}.png", dpi=300)
    plt.close()
    
    # Convert spectrograms to audio with improved function
    for i, condition in enumerate(["normal", "abnormal"]):
        # Save individual spectrogram image
        plt.figure(figsize=(10, 6))
        plt.imshow(gen_specs[i, :, :, 0], aspect='auto', origin='lower', cmap='viridis')
        plt.title(f"{condition.capitalize()} Fan Sound (Epoch {epoch})")
        plt.colorbar(format='%.2f')
        plt.xlabel('Time Frame')
        plt.ylabel('Mel Frequency Bin')
        plt.tight_layout()
        plt.savefig(f"{output_dir}/{condition}_spectrogram_epoch_{epoch}.png", dpi=300)
        plt.close()
        
        # Convert to audio using improved function with reference audio
        audio = improved_spectrogram_to_audio(gen_specs[i], reference_audio)
        sf.write(f"{output_dir}/{condition}_epoch_{epoch}.wav", audio, SR)

# 7. Main function to run the improved fan GAN training
def run_fan_gan_training(data_dir, epochs=2000, batch_size=8, samples_per_id=50):
    """Run improved GAN training specifically for fan sounds"""
    
    print("\n=== Training Fan Sound GAN ===")
    print(f"Data Directory: {data_dir}")
    print(f"Epochs: {epochs}")
    print(f"Batch Size: {batch_size}")
    print("============================\n")
    
    # Process fan data with improved preprocessing
    machine_ids = ["00", "02", "04", "06"]
    specs = []
    conditions = []
    reference_audio = None
    
    print("\nProcessing fan data with improved preprocessing...")
    machine_dir = os.path.join(data_dir, "trimmed_fan")
    
    if not os.path.exists(machine_dir):
        print(f"Error: Directory not found: {machine_dir}")
        return None, None, None
    
    # Process normal samples to get reference audio
    normal_base_dir = os.path.join(machine_dir, "normal")
    if os.path.exists(normal_base_dir):
        for machine_id in machine_ids:
            normal_id_dir = os.path.join(normal_base_dir, f"id_{machine_id}")
            if os.path.exists(normal_id_dir):
                normal_files = glob.glob(os.path.join(normal_id_dir, "*.wav"))
                
                print(f"Processing {len(normal_files)} normal files for fan id_{machine_id}")
                # Process first file to get reference audio
                if len(normal_files) > 0:
                    first_file = normal_files[0]
                    audio, _ = librosa.load(first_file, sr=SR, duration=AUDIO_LENGTH)
                    reference_audio = audio.copy()
                    
                # Process remaining files
                for file_path in tqdm(normal_files[:samples_per_id]):
                    try:
                        spec, condition, _ = load_and_preprocess_fan_audio(
                            file_path, is_abnormal=False, reference_audio=reference_audio
                        )
                        specs.append(spec)
                        conditions.append(condition)
                    except Exception as e:
                        print(f"Error processing {file_path}: {e}")
    
    # Process abnormal samples with the same reference
    abnormal_base_dir = os.path.join(machine_dir, "abnormal")
    if os.path.exists(abnormal_base_dir):
        for machine_id in machine_ids:
            abnormal_id_dir = os.path.join(abnormal_base_dir, f"id_{machine_id}")
            if os.path.exists(abnormal_id_dir):
                abnormal_files = glob.glob(os.path.join(abnormal_id_dir, "*.wav"))
                
                print(f"Processing {len(abnormal_files)} abnormal files for fan id_{machine_id}")
                for file_path in tqdm(abnormal_files[:samples_per_id]):
                    try:
                        spec, condition, _ = load_and_preprocess_fan_audio(
                            file_path, is_abnormal=True, reference_audio=reference_audio
                        )
                        specs.append(spec)
                        conditions.append(condition)
                    except Exception as e:
                        print(f"Error processing {file_path}: {e}")
    
    # Convert to numpy arrays
    specs = np.array(specs)
    conditions = np.array(conditions)
    
    print(f"Dataset shape: {specs.shape}, Conditions shape: {conditions.shape}")
    print(f"Normal samples: {np.sum(conditions == 0)}, Abnormal samples: {np.sum(conditions == 1)}")
    
    # Build fan-specific models
    generator = build_fan_generator(LATENT_DIM)
    discriminator = build_fan_discriminator()
    
    # Print model summaries
    print("\nFan Generator Summary:")
    generator.summary()
    
    print("\nFan Discriminator Summary:")
    discriminator.summary()
    
    # Build GAN
    gan = build_gan(generator, discriminator)
    
    # Compile models with improved learning rates for fan sounds
    discriminator.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.00005, beta_1=0.5),  # Slower learning
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    gan.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.00005, beta_1=0.5),  # Slower learning
        loss='binary_crossentropy'
    )
    
    # Train the fan GAN
    try:
        d_losses, g_losses = train_fan_gan(
            generator, discriminator, gan, specs, conditions, 
            epochs, batch_size, reference_audio
        )
        
        print(f"Training complete!")
        return generator, discriminator, (d_losses, g_losses)
        
    except Exception as e:
        print(f"Error during training: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, None, None

# 8. Utility function to analyze original fan sound characteristics
def analyze_fan_sound_characteristics(file_path):
    """Analyze the characteristics of an original fan sound file"""
    # Load audio
    audio, sr = librosa.load(file_path, sr=SR)
    
    # Calculate basic statistics
    rms = np.sqrt(np.mean(audio**2))
    peak = np.max(np.abs(audio))
    dynamic_range = 20 * np.log10(peak / (np.mean(np.abs(audio)) + 1e-8))
    
    # Spectral features
    spec = np.abs(librosa.stft(audio, n_fft=N_FFT, hop_length=HOP_LENGTH))
    
    # Spectral centroid (weighted mean of the frequencies)
    centroid = librosa.feature.spectral_centroid(S=spec)[0]
    
    # Find dominant frequencies (fan sounds typically have consistent tones)
    freqs = librosa.fft_frequencies(sr=SR, n_fft=N_FFT)
    spec_db = librosa.amplitude_to_db(spec)
    avg_spectrum = np.mean(spec_db, axis=1)
    
    # Plot analysis
    plt.figure(figsize=(12, 8))
    
    # Plot waveform
    plt.subplot(3, 1, 1)
    plt.plot(audio)
    plt.title('Fan Sound Waveform')
    plt.xlabel('Samples')
    plt.ylabel('Amplitude')
    plt.grid(True, alpha=0.3)
    
    # Plot spectrogram
    plt.subplot(3, 1, 2)
    plt.imshow(librosa.amplitude_to_db(spec, ref=np.max), 
              aspect='auto', origin='lower', cmap='viridis')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Fan Sound Spectrogram')
    plt.xlabel('Time Frames')
    plt.ylabel('Frequency Bins')
    
    # Plot average spectrum
    plt.subplot(3, 1, 3)
    plt.plot(freqs, avg_spectrum)
    plt.xscale('log')
    plt.grid(True, alpha=0.3)
    plt.title('Average Spectrum')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude (dB)')
    
    plt.tight_layout()
    plt.savefig('fan_sound_analysis.png', dpi=300)
    plt.close()
    
    print(f"Fan sound analysis:")
    print(f"RMS: {rms:.6f}")
    print(f"Peak amplitude: {peak:.6f}")
    print(f"Dynamic range: {dynamic_range:.2f} dB")
    print(f"Average spectral centroid: {np.mean(centroid):.2f} Hz")
    
    return {
        'audio': audio,
        'rms': rms,
        'peak': peak,
        'centroid': np.mean(centroid),
        'avg_spectrum': avg_spectrum,
        'freqs': freqs
    }

# 9. Enhanced generator training with perceptual loss for better audio quality
class FanGANTrainer:
    """Advanced trainer for fan sound GAN with perceptual losses"""
    
    def __init__(self, generator, discriminator, latent_dim, sr=16000):
        self.generator = generator
        self.discriminator = discriminator
        self.latent_dim = latent_dim
        self.sr = sr
        
        # Create GAN model
        self.discriminator.trainable = False
        self.gan = build_gan(generator, discriminator)
        
        # Create optimizer with lower learning rate for stability
        self.g_optimizer = tf.keras.optimizers.Adam(learning_rate=0.00005, beta_1=0.5)
        self.d_optimizer = tf.keras.optimizers.Adam(learning_rate=0.00005, beta_1=0.5)
        
        # Create EMA model
        self.ema_generator = tf.keras.models.clone_model(generator)
        self.ema_generator.set_weights(generator.get_weights())
        self.ema_decay = 0.999
        
        # Prepare for perceptual loss
        self.freq_weight = 1.0  # Weight for frequency-domain loss
        
    def train_step(self, real_specs, real_conditions, ref_audio=None):
        """Perform a single training step with perceptual loss"""
        batch_size = real_specs.shape[0]
        
        # -----------------
        # Train Discriminator
        # -----------------
        
        # Generate fake samples
        noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
        fake_specs = self.generator.predict([noise, real_conditions], verbose=0)
        
        # One-sided label smoothing
        real_labels = np.ones((batch_size, 1)) * 0.9 + np.random.normal(0, 0.05, (batch_size, 1))
        fake_labels = np.zeros((batch_size, 1)) + np.random.normal(0, 0.05, (batch_size, 1))
        real_labels = np.clip(real_labels, 0, 1)
        fake_labels = np.clip(fake_labels, 0, 1)
        
        # Train discriminator
        with tf.GradientTape() as tape:
            # Real samples
            real_outputs = self.discriminator([real_specs, real_conditions])
            real_loss = tf.keras.losses.binary_crossentropy(real_labels, real_outputs)
            
            # Fake samples
            fake_outputs = self.discriminator([fake_specs, real_conditions])
            fake_loss = tf.keras.losses.binary_crossentropy(fake_labels, fake_outputs)
            
            # Total loss
            d_loss = tf.reduce_mean(real_loss) + tf.reduce_mean(fake_loss)
            
        # Compute and apply gradients
        d_gradients = tape.gradient(d_loss, self.discriminator.trainable_variables)
        self.d_optimizer.apply_gradients(zip(d_gradients, self.discriminator.trainable_variables))
        
        # -----------------
        # Train Generator
        # -----------------
        
        # Generate new noise
        noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
        
        # Create balanced conditions
        random_conditions = np.zeros((batch_size, 1))
        random_conditions[batch_size//2:] = 1
        np.random.shuffle(random_conditions)
        
        # Train generator with perceptual loss
        with tf.GradientTape() as tape:
            # Generate fake samples
            fake_specs = self.generator([noise, random_conditions])
            
            # Adversarial loss
            fake_outputs = self.discriminator([fake_specs, random_conditions])
            adv_loss = tf.keras.losses.binary_crossentropy(
                np.ones((batch_size, 1)), fake_outputs
            )
            
            # Calculate perceptual loss (frequency domain)
            # This encourages spectral characteristics similar to real fan sounds
            if self.freq_weight > 0:
                # Get a batch of real fan sounds with matching conditions
                real_indices = np.where(real_conditions[:, 0] == 0)[0]
                if len(real_indices) > 0:
                    # Use a random subset of real samples for frequency matching
                    sample_indices = np.random.choice(
                        real_indices, min(4, len(real_indices)), replace=False
                    )
                    real_samples = real_specs[sample_indices]
                    
                    # Calculate average spectrum of real and fake
                    real_spec_avg = tf.reduce_mean(tf.abs(real_samples), axis=0)
                    fake_spec_avg = tf.reduce_mean(tf.abs(fake_specs), axis=0)
                    
                    # Calculate frequency loss (focus on low frequency for fan sounds)
                    # More weight on lower frequencies where fan sounds are prominent
                    weights = tf.range(1.0, float(N_MELS) + 1.0, 1.0)
                    weights = 1.0 / tf.sqrt(weights)  # More weight to lower frequencies
                    weights = tf.reshape(weights, [N_MELS, 1, 1])
                    
                    freq_diff = tf.abs(real_spec_avg - fake_spec_avg)
                    weighted_freq_diff = freq_diff * weights
                    freq_loss = tf.reduce_mean(weighted_freq_diff)
                else:
                    freq_loss = 0.0
            else:
                freq_loss = 0.0
            
            # Total loss with weighting
            g_loss = tf.reduce_mean(adv_loss) + self.freq_weight * freq_loss
            
        # Compute and apply gradients
        g_gradients = tape.gradient(g_loss, self.generator.trainable_variables)
        self.g_optimizer.apply_gradients(zip(g_gradients, self.generator.trainable_variables))
        
        # Update EMA generator
        self.update_ema()
        
        return {
            'd_loss': float(d_loss),
            'g_loss': float(g_loss),
            'freq_loss': float(freq_loss) if self.freq_weight > 0 else 0.0
        }
    
    def update_ema(self):
        """Update Exponential Moving Average of generator weights"""
        generator_weights = self.generator.get_weights()
        ema_weights = self.ema_generator.get_weights()
        
        for i in range(len(generator_weights)):
            ema_weights[i] = self.ema_decay * ema_weights[i] + (1.0 - self.ema_decay) * generator_weights[i]
            
        self.ema_generator.set_weights(ema_weights)
    
    def train(self, specs, conditions, epochs, batch_size, ref_audio=None, output_dir='fan_gan_output'):
        """Run training for specified epochs"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Arrays to store loss values
        history = {
            'd_loss': [],
            'g_loss': [],
            'freq_loss': []
        }
        
        total_batches = len(specs) // batch_size
        if total_batches == 0:
            total_batches = 1  # Handle small datasets
            
        for epoch in range(epochs):
            # Shuffle data
            indices = np.arange(len(specs))
            np.random.shuffle(indices)
            shuffled_specs = specs[indices]
            shuffled_conditions = conditions[indices].reshape(-1, 1)
            
            # Train on batches
            epoch_d_loss = 0
            epoch_g_loss = 0
            epoch_freq_loss = 0
            
            for batch in range(total_batches):
                start_idx = batch * batch_size
                end_idx = min((batch + 1) * batch_size, len(specs))
                
                batch_specs = shuffled_specs[start_idx:end_idx]
                batch_conditions = shuffled_conditions[start_idx:end_idx]
                
                # Perform training step
                step_losses = self.train_step(batch_specs, batch_conditions, ref_audio)
                
                # Accumulate losses
                epoch_d_loss += step_losses['d_loss']
                epoch_g_loss += step_losses['g_loss']
                epoch_freq_loss += step_losses['freq_loss']
            
            # Calculate average losses
            epoch_d_loss /= total_batches
            epoch_g_loss /= total_batches
            epoch_freq_loss /= total_batches
            
            # Store losses
            history['d_loss'].append(epoch_d_loss)
            history['g_loss'].append(epoch_g_loss)
            history['freq_loss'].append(epoch_freq_loss)
            
            # Print progress
            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"[Fan] Epoch {epoch+1}/{epochs} | D Loss: {epoch_d_loss:.4f} | "
                      f"G Loss: {epoch_g_loss:.4f} | Freq Loss: {epoch_freq_loss:.4f}")
            
            # Save samples at specific milestones
            if (epoch + 1) in [1, 100, 500, 1000, 2000] or (epoch + 1) == epochs:
                print(f"\nGenerating samples for epoch {epoch+1}...")
                self.save_samples(epoch + 1, output_dir, ref_audio)
        
        # Save final models
        self.generator.save(f"{output_dir}/fan_generator.h5")
        self.discriminator.save(f"{output_dir}/fan_discriminator.h5")
        self.ema_generator.save(f"{output_dir}/fan_ema_generator.h5")
        
        # Plot loss curves
        plt.figure(figsize=(10, 6))
        plt.plot(history['d_loss'], label='Discriminator')
        plt.plot(history['g_loss'], label='Generator')
        if self.freq_weight > 0:
            plt.plot(history['freq_loss'], label='Frequency Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Fan GAN Training Loss')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.savefig(f"{output_dir}/training_loss.png", dpi=300)
        plt.close()
        
        return history
    
    def save_samples(self, epoch, output_dir, ref_audio=None):
        """Save generated samples using EMA generator"""
        # Generate noise and conditions
        noise = np.random.normal(0, 1, (2, self.latent_dim))
        conditions = np.array([[0], [1]])  # Normal and abnormal
        
        # Generate spectrograms using EMA generator for better quality
        gen_specs = self.ema_generator.predict([noise, conditions], verbose=0)
        
        # Save spectrograms
        plt.figure(figsize=(12, 6))
        
        plt.subplot(1, 2, 1)
        plt.imshow(gen_specs[0, :, :, 0], aspect='auto', origin='lower', cmap='viridis')
        plt.title("Normal Fan Sound")
        plt.colorbar(format='%.2f')
        plt.xlabel('Time Frame')
        plt.ylabel('Mel Frequency Bin')
        
        plt.subplot(1, 2, 2)
        plt.imshow(gen_specs[1, :, :, 0], aspect='auto', origin='lower', cmap='viridis')
        plt.title("Abnormal Fan Sound")
        plt.colorbar(format='%.2f')
        plt.xlabel('Time Frame')
        plt.ylabel('Mel Frequency Bin')
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/spectrograms_epoch_{epoch}.png", dpi=300)
        plt.close()
        
        # Convert to audio with improved function
        for i, condition in enumerate(["normal", "abnormal"]):
            # Convert with improved function using reference
            audio = improved_spectrogram_to_audio(gen_specs[i], ref_audio)
            sf.write(f"{output_dir}/{condition}_epoch_{epoch}.wav", audio, self.sr)
            
            # Also save the spectrogram used
            plt.figure(figsize=(10, 6))
            plt.imshow(gen_specs[i, :, :, 0], aspect='auto', origin='lower', cmap='viridis')
            plt.title(f"{condition.capitalize()} Fan Sound (Epoch {epoch})")
            plt.colorbar(format='%.2f')
            plt.tight_layout()
            plt.savefig(f"{output_dir}/{condition}_spectrogram_{epoch}.png", dpi=300)
            plt.close()

# 10. Complete pipeline for training and producing improved fan sounds
def run_improved_fan_gan_pipeline(data_dir, epochs=2000, batch_size=8):
    """Full pipeline for training improved fan sound generation"""
    
    # Step 1: Analyze original fan sounds to understand their characteristics
    print("\n=== Analyzing Original Fan Sounds ===")
    reference_audio = None
    reference_stats = None
    
    # Find a normal fan sound file to use as reference
    fan_dir = os.path.join(data_dir, "trimmed_fan", "normal")
    if os.path.exists(fan_dir):
        for machine_id in ["00", "02", "04", "06"]:
            id_dir = os.path.join(fan_dir, f"id_{machine_id}")
            if os.path.exists(id_dir):
                files = glob.glob(os.path.join(id_dir, "*.wav"))
                if files:
                    # Analyze the first file
                    reference_file = files[0]
                    print(f"Using reference file: {reference_file}")
                    reference_stats = analyze_fan_sound_characteristics(reference_file)
                    reference_audio = reference_stats['audio']
                    break
    
    if reference_audio is None:
        print("Warning: No reference audio found. Using default parameters.")
    else:
        print(f"Reference audio RMS: {reference_stats['rms']:.6f}")
        print(f"Reference audio centroid: {reference_stats['centroid']:.2f} Hz")
    
    # Step 2: Process the fan data with optimized preprocessing
    print("\n=== Processing Fan Data ===")
    specs, conditions, reference_samples = process_fan_data(data_dir, reference_audio)
    
    if len(specs) == 0:
        print("Error: No fan data found.")
        return None
    
    # Step 3: Build improved fan generator and discriminator
    print("\n=== Building Fan GAN Models ===")
    generator = build_fan_generator(LATENT_DIM)
    discriminator = build_fan_discriminator()
    
    # Step 4: Setup the advanced trainer
    trainer = FanGANTrainer(generator, discriminator, LATENT_DIM, SR)
    
    # Adjust frequency weight based on reference analysis if available
    if reference_stats is not None:
        # Emphasize frequency loss more for quieter fans (which are harder to model)
        if reference_stats['rms'] < 0.05:
            trainer.freq_weight = 2.0
            print("Quiet fan detected: Increasing frequency loss weight to 2.0")
    
    # Step 5: Run training with advanced techniques
    print("\n=== Training Fan GAN with Advanced Techniques ===")
    output_dir = "improved_fan_sounds"
    history = trainer.train(specs, conditions, epochs, batch_size, reference_audio, output_dir)
    
    # Step 6: Generate final samples with perceptual enhancements
    print("\n=== Generating Final Fan Sound Samples ===")
    generate_enhanced_fan_samples(trainer.ema_generator, output_dir, reference_audio)
    
    # Step 7: Compare original and generated sounds
    print("\n=== Comparing Original and Generated Fan Sounds ===")
    compare_original_and_generated(reference_stats, trainer.ema_generator, output_dir)
    
    return trainer.ema_generator

# Helper function to process fan data with optimized preprocessing
def process_fan_data(data_dir, reference_audio=None):
    """Process fan data with optimized preprocessing"""
    specs = []
    conditions = []
    reference_samples = []
    
    machine_dir = os.path.join(data_dir, "trimmed_fan")
    
    if not os.path.exists(machine_dir):
        print(f"Error: Directory not found: {machine_dir}")
        return np.array([]), np.array([]), reference_samples
    
    # Process normal samples
    normal_base_dir = os.path.join(machine_dir, "normal")
    if os.path.exists(normal_base_dir):
        for machine_id in ["00", "02", "04", "06"]:
            normal_id_dir = os.path.join(normal_base_dir, f"id_{machine_id}")
            if os.path.exists(normal_id_dir):
                normal_files = glob.glob(os.path.join(normal_id_dir, "*.wav"))
                
                print(f"Processing {len(normal_files)} normal files for fan id_{machine_id}")
                for file_path in tqdm(normal_files[:50]):
                    try:
                        # Load and preprocess with fan-specific function
                        spec, condition, ref = load_and_preprocess_fan_audio(
                            file_path, is_abnormal=False, reference_audio=reference_audio
                        )
                        specs.append(spec)
                        conditions.append(condition)
                        
                        # Store reference audio from first few samples
                        if len(reference_samples) < 5:
                            audio, _ = librosa.load(file_path, sr=SR, duration=AUDIO_LENGTH)
                            reference_samples.append(audio)
                    except Exception as e:
                        print(f"Error processing {file_path}: {e}")
    
    # Process abnormal samples
    abnormal_base_dir = os.path.join(machine_dir, "abnormal")
    if os.path.exists(abnormal_base_dir):
        for machine_id in ["00", "02", "04", "06"]:
            abnormal_id_dir = os.path.join(abnormal_base_dir, f"id_{machine_id}")
            if os.path.exists(abnormal_id_dir):
                abnormal_files = glob.glob(os.path.join(abnormal_id_dir, "*.wav"))
                
                print(f"Processing {len(abnormal_files)} abnormal files for fan id_{machine_id}")
                for file_path in tqdm(abnormal_files[:50]):
                    try:
                        # Load and preprocess
                        spec, condition, _ = load_and_preprocess_fan_audio(
                            file_path, is_abnormal=True, reference_audio=reference_audio
                        )
                        specs.append(spec)
                        conditions.append(condition)
                    except Exception as e:
                        print(f"Error processing {file_path}: {e}")
    
    # Convert to numpy arrays
    specs = np.array(specs)
    conditions = np.array(conditions)
    
    print(f"Dataset shape: {specs.shape}, Conditions shape: {conditions.shape}")
    print(f"Normal samples: {np.sum(conditions == 0)}, Abnormal samples: {np.sum(conditions == 1)}")
    
    return specs, conditions, reference_samples

# Generate enhanced fan samples with perceptual improvements
def generate_enhanced_fan_samples(generator, output_dir, reference_audio, num_samples=10):
    """Generate fan sound samples with perceptual enhancements"""
    
    final_dir = os.path.join(output_dir, "final_samples")
    os.makedirs(final_dir, exist_ok=True)
    
    for condition_label, condition_name in enumerate(["normal", "abnormal"]):
        condition_dir = os.path.join(final_dir, condition_name)
        os.makedirs(condition_dir, exist_ok=True)
        
        print(f"Generating {num_samples} {condition_name} fan samples...")
        for i in range(num_samples):
            # Generate with added variations in the latent space for diversity
            noise = np.random.normal(0, 1, (1, LATENT_DIM))
            
            # Add some oscillatory pattern for fan-like periodicity
            # Fans typically have consistent cyclical sounds
            time_steps = 100
            t = np.linspace(0, 2*np.pi, time_steps)
            oscillation = np.sin(t) * 0.2  # Subtle oscillation
            
            # Add oscillation to a subset of latent dimensions (first 10)
            for j in range(min(10, LATENT_DIM)):
                noise[0, j] += oscillation[j % time_steps]
            
            condition = np.array([[condition_label]])
            
            # Generate spectrogram
            gen_spec = generator.predict([noise, condition], verbose=0)
            
            # Convert to audio with enhanced conversion
            audio = improved_spectrogram_to_audio(gen_spec[0], reference_audio)
            
            # Save audio with additional post-processing for fan characteristics
            # Apply subtle repetitive pattern enhancement for fan-like regularity
            if condition_label == 0:  # Normal fan sound
                # Emphasis consistent periodic nature of fans
                # Find peaks in the audio that might represent fan rotation
                from scipy.signal import find_peaks
                peaks, _ = find_peaks(audio, distance=int(SR*0.05))  # Look for peaks at least 50ms apart
                
                if len(peaks) > 3:
                    # Calculate median distance between peaks (fan rotation period)
                    peak_distances = np.diff(peaks)
                    median_distance = np.median(peak_distances)
                    
                    # Enhance periodicity slightly by applying gentle comb filter at fan frequency
                    if 0.01 < median_distance / SR < 0.5:  # Between 2-100 Hz (typical fan speeds)
                        fan_freq = SR / median_distance
                        print(f"Sample {i+1}: Detected fan frequency ~{fan_freq:.1f} Hz, enhancing periodicity")
                        
                        # Apply subtle resonance at fan frequency
                        def apply_comb_filter(audio, delay_samples, feedback=0.2):
                            output = audio.copy()
                            for i in range(delay_samples, len(audio)):
                                output[i] += audio[i - delay_samples] * feedback
                            return output
                        
                        # Apply very subtle comb filter at detected fan frequency
                        audio = apply_comb_filter(audio, int(median_distance), feedback=0.15)
            
            # Normalize final output
            audio = audio / (np.max(np.abs(audio)) + 1e-8) * 0.8
            
            # Save the final processed audio
            sf.write(f"{condition_dir}/{condition_name}_sample_{i+1}.wav", audio, SR)
            
            # Save spectrogram visualization
            plt.figure(figsize=(10, 6))
            plt.imshow(gen_spec[0, :, :, 0], aspect='auto', origin='lower', cmap='viridis')
            plt.title(f"{condition_name.capitalize()} Fan Sound Sample {i+1}")
            plt.colorbar(format='%.2f')
            plt.tight_layout()
            plt.savefig(f"{condition_dir}/{condition_name}_sample_{i+1}.png", dpi=300)
            plt.close()
    
    print(f"Enhanced fan samples saved in '{final_dir}'")

# Compare original and generated fan sounds
def compare_original_and_generated(reference_stats, generator, output_dir):
    """Create detailed comparison between original and generated fan sounds"""
    
    if reference_stats is None:
        print("No reference stats available for comparison.")
        return
    
    comparison_dir = os.path.join(output_dir, "comparison")
    os.makedirs(comparison_dir, exist_ok=True)
    
    # Generate a normal fan sound
    noise = np.random.normal(0, 1, (1, LATENT_DIM))
    condition = np.array([[0]])  # Normal condition
    gen_spec = generator.predict([noise, condition], verbose=0)
    gen_audio = improved_spectrogram_to_audio(gen_spec[0], reference_stats['audio'])
    
    # Calculate generated audio stats
    gen_rms = np.sqrt(np.mean(gen_audio**2))
    gen_peak = np.max(np.abs(gen_audio))
    
    # Calculate spectral features
    gen_spec_stft = np.abs(librosa.stft(gen_audio, n_fft=N_FFT, hop_length=HOP_LENGTH))
    gen_centroid = librosa.feature.spectral_centroid(S=gen_spec_stft)[0]
    
    # Create comparison plots
    plt.figure(figsize=(15, 10))
    
    # Compare waveforms
    plt.subplot(3, 2, 1)
    plt.plot(reference_stats['audio'])
    plt.title('Original Fan Sound Waveform')
    plt.xlabel('Samples')
    plt.ylabel('Amplitude')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(3, 2, 2)
    plt.plot(gen_audio)
    plt.title('Generated Fan Sound Waveform')
    plt.xlabel('Samples')
    plt.ylabel('Amplitude')
    plt.grid(True, alpha=0.3)
    
    # Compare spectrograms
    plt.subplot(3, 2, 3)
    ref_spec = librosa.amplitude_to_db(
        np.abs(librosa.stft(reference_stats['audio'], n_fft=N_FFT, hop_length=HOP_LENGTH))
    )
    plt.imshow(ref_spec, aspect='auto', origin='lower', cmap='viridis')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Original Fan Sound Spectrogram')
    plt.xlabel('Time Frames')
    plt.ylabel('Frequency Bins')
    
    plt.subplot(3, 2, 4)
    gen_spec_db = librosa.amplitude_to_db(gen_spec_stft)
    plt.imshow(gen_spec_db, aspect='auto', origin='lower', cmap='viridis')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Generated Fan Sound Spectrogram')
    plt.xlabel('Time Frames')
    plt.ylabel('Frequency Bins')
    
    # Compare average spectrum
    plt.subplot(3, 2, 5)
    plt.plot(reference_stats['freqs'], reference_stats['avg_spectrum'], label='Original')
    
    gen_avg_spectrum = np.mean(librosa.amplitude_to_db(gen_spec_stft), axis=1)
    plt.plot(reference_stats['freqs'], gen_avg_spectrum, label='Generated')
    
    plt.xscale('log')
    plt.grid(True, alpha=0.3)
    plt.title('Average Spectrum Comparison')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude (dB)')
    plt.legend()
    
    # Add text comparison of stats
    plt.subplot(3, 2, 6)
    plt.axis('off')
    
    stats_text = (
        f"COMPARISON STATS:\n\n"
        f"Original Fan Sound:\n"
        f"- RMS Amplitude: {reference_stats['rms']:.6f}\n"
        f"- Peak Amplitude: {reference_stats['peak']:.6f}\n"
        f"- Spectral Centroid: {reference_stats['centroid']:.2f} Hz\n\n"
        f"Generated Fan Sound:\n"
        f"- RMS Amplitude: {gen_rms:.6f}\n"
        f"- Peak Amplitude: {gen_peak:.6f}\n"
        f"- Spectral Centroid: {np.mean(gen_centroid):.2f} Hz\n\n"
        f"Difference:\n"
        f"- RMS Ratio: {gen_rms/reference_stats['rms']:.2f}x\n"
        f"- Centroid Ratio: {np.mean(gen_centroid)/reference_stats['centroid']:.2f}x"
    )
    
    plt.text(0.1, 0.5, stats_text, fontsize=12, va='center')
    
    plt.tight_layout()



# Method 1: Using the main() function directly in your code

# Replace 'your_data_directory' with the path to your MIMII dataset
data_dir = "C:/Users/julia/DS4440/machine-noise-generator/data"  # Replace with your actual path

# Choose the method - 'perceptual' generally gives the best results
# Options: 'standard', 'perceptual', or 'wgan'
method = 'perceptual'  

# Run with default epochs (1000) and batch size (8)
main(data_dir, method=method)

# Or specify custom training parameters
# main(data_dir, method='perceptual', epochs=2000, batch_size=16)


# Method 2: Running as a script from command line
# Save the code I provided as 'improved_fan_gan.py', then run:
"""
python improved_fan_gan.py --data_dir /path/to/mimii_dataset/ --method perceptual --epochs 1000 --batch_size 8
"""


# Method 3: Using just the post-processing on your existing model
# If you've already trained a model and just want to improve the audio conversion:

# First, load your existing generator model
