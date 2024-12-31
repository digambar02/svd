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

    # Display brief theory
    st.write("""
    **Singular Value Decomposition (SVD)** decomposes a matrix A into three matrices: A = U × S × Vᵀ
    
    For an image of size (m × n):
    - **U**: Left singular vectors (m × k matrix)
    - **S**: Singular values (k × k diagonal matrix)
    - **Vᵀ**: Right singular vectors (k × n matrix)
    
    where k is the number of components chosen for compression.
    """)

    uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

    if uploaded_file is not None:
        try:
            image = Image.open(uploaded_file)
            
            # Calculate maximum k value
            max_k = min(image.size[0], image.size[1])
            default_k = min(50, max_k)

            # Slider for k
            k = st.slider("Choose the number of components (k)", 
                         min_value=1, 
                         max_value=max_k,
                         value=default_k)

            # Compress image and get components
            compressed_img, original_img, components = svd_compress(image, k)

            # Display original and compressed images with dimensions
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Original Image")
                st.image(image, use_container_width=True)
                st.write(f"Dimensions: {format_matrix_shape(components['shapes']['original'])}")
            with col2:
                st.subheader("Compressed Image")
                st.image(compressed_img, use_container_width=True)
                st.write(f"Dimensions: {format_matrix_shape(compressed_img.shape)}")
            
            # Create compressed image buffer
            compressed_img_pil = Image.fromarray(compressed_img)
            compressed_buffer = io.BytesIO()
            compressed_img_pil.save(compressed_buffer, format="PNG", optimize=True)
            compressed_buffer.seek(0)

            # Download button
            st.download_button(
                label="Download Compressed Image",
                data=compressed_buffer,
                file_name="compressed_image.png",
                mime="image/png"
            )

            # Calculate actual file sizes
            st.write("---")
            st.subheader("File Size Comparison")
            
            # Get original file size
            uploaded_file.seek(0)
            original_bytes = uploaded_file.read()
            original_size = len(original_bytes) / (1024 * 1024)  # Convert to MB
            
            # Get compressed file size
            compressed_size = len(compressed_buffer.getvalue()) / (1024 * 1024)  # Convert to MB
            
            # Display size metrics
            # col3, col4, col5 = st.columns(3)
            # with col3:
            #     st.metric("Original File Size", f"{original_size:.2f} MB")
            # with col4:
            #     st.metric("Compressed File Size", f"{compressed_size:.2f} MB")
            # with col5:
            #     reduction = 100 * (1 - compressed_size / original_size)
            #     st.metric("Size Reduction", f"{reduction:.1f}%")

            # Matrix multiplication explanation
            st.write("---")
            st.write("**Matrix Multiplication:**")
            st.latex(r"""
            \text{Image}_{(m \times n)} = U_{(m \times k)} \times S_{(k \times k)} \times V^T_{(k \times n)}
            """)

            # Display component matrices with dimensions
            st.subheader("SVD Components Visualization")
            comp_col1, comp_col2, comp_col3 = st.columns(3)
            
            with comp_col1:
                st.write("**U Matrix** (Left Singular Vectors)")
                u_vis = visualize_matrix(components['U'], "U")
                st.image(u_vis, use_container_width=True)
                st.write(f"Dimensions: {format_matrix_shape(components['shapes']['U'])}")
            
            with comp_col2:
                st.write("**S Matrix** (Singular Values)")
                s_vis = visualize_matrix(components['S'], "S")
                st.image(s_vis, use_container_width=True)
                st.write(f"Dimensions: {format_matrix_shape(components['shapes']['S'])}")
            
            with comp_col3:
                st.write("**Vt Matrix** (Right Singular Vectors)")
                vt_vis = visualize_matrix(components['Vt'], "Vt")
                st.image(vt_vis, use_container_width=True)
                st.write(f"Dimensions: {format_matrix_shape(components['shapes']['Vt'])}")

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            st.error("Please try uploading a different image or adjusting the compression settings.")

if __name__ == "__main__":
    main()