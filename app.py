import streamlit as st
import numpy as np
from PIL import Image
import io

# Set page configuration
st.set_page_config(page_title="Image SVD Compression", layout="centered")

def visualize_matrix(matrix, title):
    """Helper function to visualize matrices as images"""
    # Normalize the matrix to 0-255 range for visualization
    if matrix.ndim == 2:
        normalized = ((matrix - matrix.min()) * (255.0 / (matrix.max() - matrix.min()))).astype(np.uint8)
    else:
        # For RGB images, normalize each channel separately
        normalized = np.zeros_like(matrix, dtype=np.uint8)
        for i in range(3):
            channel = matrix[:,:,i]
            normalized[:,:,i] = ((channel - channel.min()) * (255.0 / (channel.max() - channel.min()))).astype(np.uint8)
    return normalized

def format_matrix_shape(shape):
    """Format matrix shape into a readable string"""
    if len(shape) == 3:
        return f"{shape[0]} × {shape[1]} × {shape[2]} (H × W × RGB)"
    else:
        return f"{shape[0]} × {shape[1]} (H × W)"

def svd_compress(image, k):
    # Convert image to numpy array
    img_array = np.array(image, dtype=np.float32)
    
    # Store component matrices and their shapes for visualization
    components = {'U': [], 'S': [], 'Vt': [], 'shapes': {}}
    
    if len(img_array.shape) == 3:
        # Process each color channel separately for RGB images
        compressed_channels = []
        for channel in range(3):
            U, S, Vt = np.linalg.svd(img_array[:,:,channel], full_matrices=False)
            
            # Store components for this channel
            components['U'].append(U[:, :k])
            components['S'].append(np.diag(S[:k]))
            components['Vt'].append(Vt[:k, :])
            
            # Compute compressed channel
            compressed_channel = np.dot(U[:, :k], np.multiply(S[:k, None], Vt[:k, :]))
            compressed_channels.append(compressed_channel)
            
        compressed_img = np.stack(compressed_channels, axis=2)
        
        # Stack channels for visualization
        components['U'] = np.stack(components['U'], axis=2)
        components['S'] = np.stack(components['S'], axis=2)
        components['Vt'] = np.stack(components['Vt'], axis=2)
        
    else:
        # Process grayscale images
        U, S, Vt = np.linalg.svd(img_array, full_matrices=False)
        
        # Store components
        components['U'] = U[:, :k]
        components['S'] = np.diag(S[:k])
        components['Vt'] = Vt[:k, :]
    
    # Store shapes
    components['shapes'] = {
        'original': img_array.shape,
        'U': components['U'].shape,
        'S': components['S'].shape,
        'Vt': components['Vt'].shape
    }
    
    # Clip values and convert to uint8
    compressed_img = np.clip(compressed_img, 0, 255).astype(np.uint8)
    
    return compressed_img, img_array, components

def main():
    st.title('Image Compression using SVD')
    uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

    if uploaded_file is not None:
        try:
            image = Image.open(uploaded_file)
            original_format = image.format
            
            # Compression controls
            col1, col2 = st.columns(2)
            with col1:
                max_k = min(image.size[0], image.size[1])
                k = st.slider("SVD Components (k)", 1, max_k, min(50, max_k))
            with col2:
                quality = st.slider("Output Quality", 1, 100, 85)

            # Compress image
            compressed_img, original_img, _ = svd_compress(image, k)
            compressed_img_pil = Image.fromarray(compressed_img)

            # Save compressed image
            compressed_buffer = io.BytesIO()
            if original_format == 'JPEG':
                compressed_img_pil.save(compressed_buffer, 
                                     format="JPEG", 
                                     quality=quality, 
                                     optimize=True,
                                     progressive=True)
            else:
                compressed_img_pil.save(compressed_buffer, 
                                     format="PNG", 
                                     optimize=True,
                                     compression_level=9)
            
            # Display images and metrics
            col1, col2 = st.columns(2)
            with col1:
                st.image(image, caption="Original")
            with col2:
                st.image(compressed_img, caption="Compressed")

            # Calculate sizes
            uploaded_file.seek(0)
            compressed_buffer.seek(0)
            original_size = len(uploaded_file.read()) / 1024  # KB
            compressed_size = len(compressed_buffer.getvalue()) / 1024  # KB
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Original", f"{original_size:.1f} KB")
            with col2:
                st.metric("Compressed", f"{compressed_size:.1f} KB")
            with col3:
                reduction = 100 * (1 - compressed_size/original_size)
                st.metric("Reduction", f"{reduction:.1f}%")

            # Download button
            st.download_button(
                "Download Compressed Image",
                compressed_buffer,
                f"compressed.{original_format.lower()}",
                f"image/{original_format.lower()}"
            )

        except Exception as e:
            st.error(f"Error: {str(e)}")
            
if __name__ == "__main__":
    main()