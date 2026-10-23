import pandas as pd
import random
import os
import numpy as np
import OpenEXR
import Imath
from PIL import Image
import matplotlib.pyplot as plt
import plyfile
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
import time
import matplotlib
matplotlib.use('TkAgg')  # 使用TkAgg后端支持交互

def read_and_convert_csv(csv_path):
    """
    Read CSV file and convert path formats
    
    Parameters:
    - csv_path: Path to CSV file
    
    Returns:
    - DataFrame with converted paths
    """
    # Try different encodings to read CSV file
    try:
        # First try UTF-8 encoding with comma separator
        df = pd.read_csv(csv_path, sep=',', encoding='utf-8')
    except Exception as utf8_error:
        try:
            # If UTF-8 fails, try GBK encoding
            df = pd.read_csv(csv_path, sep=',', encoding='gbk')
        except Exception as gbk_error:
            try:
                # Try reading with space separator
                df = pd.read_csv(csv_path, sep=' ', encoding='utf-8')
            except Exception as space_error:
                try:
                    # Try reading with space separator and GBK encoding
                    df = pd.read_csv(csv_path, sep=' ', encoding='gbk')
                except Exception as e:
                    print(f"Unable to read CSV file: {e}")
                    return None
    
    # Print CSV column names for debugging
    print("CSV file column names:", df.columns.tolist())
    
    # Standardize column names (remove spaces, convert to lowercase)
    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
    print("Standardized column names:", df.columns.tolist())
    
    # Check if required columns exist
    required_columns = ['scan_path', 'voxel_path', 'full_feat_path', 'render_dir']
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        print(f"Error: CSV file is missing required columns: {missing_columns}")
        return None
    
    # Base path (parent directory of CSV file)
    base_dir = os.path.dirname(csv_path)
    project_root = os.path.dirname(base_dir)
    
    print(f"Base directory: {base_dir}")
    print(f"Project root directory: {project_root}")
    
    # Convert path formats
    converted_data = []
    
    for index, row in df.iterrows():
        # Get original paths
        scan_path_relative = row['scan_path']
        voxel_path_relative = row['voxel_path']
        feat_path_relative = row['full_feat_path']
        render_dir_relative = row['render_dir']
        
        # Convert path formats
        scan_file = os.path.join(base_dir, scan_path_relative) #.replace('/', '\\'))
        voxel_file = os.path.join(base_dir, voxel_path_relative) #.replace('/', '\\'))
        feature_file = os.path.join(base_dir, feat_path_relative) #.replace('/', '\\'))
        exr_source_folder = os.path.join(base_dir, render_dir_relative) #.replace('/', '\\'))
        exr_source_folder = base_dir + exr_source_folder
        # Create converted data row
        converted_row = {
            'terrain_src': row.get('terrain_src', ''),
            'terrain_tag': row.get('terrain_tag', ''),
            'patch_file': row.get('patch_file', ''),
            'patch_id': row.get('patch_id', ''),
            'scan_id': row.get('scan_id', ''),
            'scan_path': scan_path_relative,
            'scan_file': scan_file,
            'voxel_id': row.get('voxel_id', ''),
            'voxel_path': voxel_path_relative,
            'voxel_file': voxel_file,
            'feat_id': row.get('feat_id', ''),
            'full_feat_path': feat_path_relative,
            'feature_file': feature_file,
            'render_subdir_id': row.get('render_subdir_id', ''),
            'render_dir': render_dir_relative,
            'exr_source_folder': exr_source_folder,
            'lidar_height': row.get('lidar_height', 0),
            'scale_factor': row.get('scale_factor', 1.0),
            'random_pose_rate': row.get('random_pose_rate', 0),
            'blender_path': row.get('blender_path', ''),
            'toolkit': row.get('toolkit', ''),
            'seed': row.get('seed', 0),
            'timestamp': row.get('timestamp', '')
        }
        
        converted_data.append(converted_row)
    
    return pd.DataFrame(converted_data)

def exr_to_png(exr_path, png_path=None, exposure=1.0, gamma=2.2, normalize=True):
    """
    Convert EXR file to PNG format
    
    Parameters:
    - exr_path: Path to EXR file
    - png_path: Output PNG file path (if None, auto-generated)
    - exposure: Exposure adjustment factor
    - gamma: Gamma correction value
    - normalize: Whether to automatically normalize brightness range
    """
    
    # Auto-generate output path if not specified
    if png_path is None:
        base_name = os.path.splitext(exr_path)[0]
        png_path = base_name + '.png'
    
    try:
        # 1. Read EXR file
        exr_file = OpenEXR.InputFile(exr_path)
        
        # Get image dimensions
        header = exr_file.header()
        dw = header['dataWindow']
        width = dw.max.x - dw.min.x + 1
        height = dw.max.y - dw.min.y + 1
        
        # 2. Get all available channel names
        available_channels = list(header['channels'].keys())
        
        # 3. Find appropriate channels
        # Prioritize standard RGB channels
        channel_mapping = {
            'r': None,
            'g': None,
            'b': None
        }
        
        # Try to match various possible channel names
        for channel in available_channels:
            lower_channel = channel.lower()
            
            if 'r' in lower_channel or 'red' in lower_channel or 'x' in lower_channel:
                channel_mapping['r'] = channel
            elif 'g' in lower_channel or 'green' in lower_channel or 'y' in lower_channel:
                channel_mapping['g'] = channel
            elif 'b' in lower_channel or 'blue' in lower_channel or 'z' in lower_channel:
                channel_mapping['b'] = channel
        
        # If standard channels not found, use first channel as grayscale
        if channel_mapping['r'] is None:
            print(f"Warning: {os.path.basename(exr_path)} - Standard RGB channels not found, using first channel as grayscale")
            first_channel = available_channels[0]
            channel_mapping['r'] = first_channel
            channel_mapping['g'] = first_channel
            channel_mapping['b'] = first_channel
        
        # 4. Read channel data
        pt = Imath.PixelType(Imath.PixelType.FLOAT)
        
        # Read selected channels
        red_str = exr_file.channel(channel_mapping['r'], pt)
        green_str = exr_file.channel(channel_mapping['g'], pt)
        blue_str = exr_file.channel(channel_mapping['b'], pt)
        
        # Convert string data to numpy arrays
        red = np.frombuffer(red_str, dtype=np.float32).reshape(height, width)
        green = np.frombuffer(green_str, dtype=np.float32).reshape(height, width)
        blue = np.frombuffer(blue_str, dtype=np.float32).reshape(height, width)
        
        # 5. Merge channels to create RGB image
        img_linear = np.stack([red, green, blue], axis=2)
        
        # 6. Apply exposure adjustment
        img_exposed = img_linear * exposure
        
        # 7. Tone mapping and gamma correction
        if normalize:
            # Auto-normalize to 0-1 range (preserving top 0.1% outliers)
            v_min = np.percentile(img_exposed, 0.1)
            v_max = np.percentile(img_exposed, 99.9)
            img_mapped = (img_exposed - v_min) / (v_max - v_min)
            img_mapped = np.clip(img_mapped, 0, 1)
        else:
            # Simple Reinhard tone mapping
            img_mapped = img_exposed / (1.0 + img_exposed)
        
        # 8. Apply gamma correction (linear to sRGB)
        img_srgb = np.power(img_mapped, 1.0/gamma)
        
        # 9. Convert to 8-bit integer (0-255)
        img_8bit = (img_srgb * 255).astype(np.uint8)
        
        # 10. Create PIL image and save as PNG
        pil_img = Image.fromarray(img_8bit)
        pil_img.save(png_path, 'PNG')
        
        print(f"Successfully converted: {os.path.basename(exr_path)} -> {os.path.basename(png_path)}")
        
        return png_path
        
    except Exception as e:
        print(f"Conversion failed {os.path.basename(exr_path)}: {str(e)}")
        return None

def convert_and_display_random_12(source_folder, exposure=1.5, gamma=2.2, normalize=True):
    """
    Randomly select 12 EXR files, convert to PNG, and display in 3×4 grid
    
    Parameters:
    - source_folder: Path to folder containing EXR files
    - exposure: Exposure adjustment factor
    - gamma: Gamma correction value
    - normalize: Whether to automatically normalize brightness range
    
    Returns:
    - Matplotlib figure object
    """
    print("FUCK!!")
    # Check if source folder exists
    if not os.path.exists(source_folder):
        print(f"Source folder does not exist: {source_folder}")
        return None
    
    # Get parent directory of source folder
    parent_dir = os.path.dirname(source_folder)
    
    # Get source folder name
    folder_name = os.path.basename(source_folder)
    
    # Create target folder name (append "_png" to original name)
    target_folder_name = folder_name + "_png"
    
    # Create target folder path
    target_folder = os.path.join(parent_dir, target_folder_name)
    
    # Create target folder if it doesn't exist
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)
        print(f"Created target folder: {target_folder}")
    
    # Get all EXR files
    exr_files = [f for f in os.listdir(source_folder) if f.lower().endswith('.exr')]
    
    if not exr_files:
        print(f"No EXR files found in folder: {source_folder}")
        return None
    
    # Check number of files
    if len(exr_files) < 12:
        print(f"Warning: Only {len(exr_files)} EXR files found, need at least 12")
        print("Using all available files")
        selected_files = exr_files
    else:
        # Randomly select 12 files
        selected_files = random.sample(exr_files, 12)
    
    print(f"Randomly selected {len(selected_files)} EXR files for conversion and display...")
    
    # Convert selected files
    converted_files = []
    for exr_file in selected_files:
        exr_path = os.path.join(source_folder, exr_file)
        png_file = os.path.splitext(exr_file)[0] + '.png'
        png_path = os.path.join(target_folder, png_file)
        
        result = exr_to_png(
            exr_path, 
            png_path=png_path,
            exposure=exposure,
            gamma=gamma,
            normalize=normalize
        )
        
        if result:
            converted_files.append(result)
    
    # Check number of successful conversions
    if not converted_files:
        print("No files converted successfully, cannot display")
        return None
    
    # Create figure for EXR images
    fig_exr = plt.figure(figsize=(15, 12))
    fig_exr.suptitle(f"Randomly Selected EXR Images ({len(converted_files)} files)", fontsize=16)
    
    # Create 3×4 subplot grid
    rows = 3
    cols = 4
    axes = fig_exr.subplots(rows, cols)
    fig_exr.subplots_adjust(wspace=0.1, hspace=0.1)  # Adjust subplot spacing
    
    # Adjust grid layout if fewer than 12 files
    if len(converted_files) < rows * cols:
        # Calculate actual rows needed
        actual_rows = (len(converted_files) + cols - 1) // cols
        # Remove extra subplots
        for i in range(len(converted_files), rows*cols):
            fig_exr.delaxes(axes.flatten()[i])
    
    # Display each image
    for i, ax in enumerate(axes.flatten()):
        if i < len(converted_files):
            img_path = converted_files[i]
            img = Image.open(img_path)
            ax.imshow(img)
            ax.axis('off')  # Turn off axes
            ax.set_title(f"Image {i+1}", fontsize=10)
        else:
            ax.axis('off')  # Hide extra subplots
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Make space for main title
    
    print(f"Conversion and display completed for {len(converted_files)} images")
    print(f"All PNG files saved to: {target_folder}")
    
    return fig_exr

def visualize_3d_data(scan_file, voxel_file, feature_file):
    """
    Visualize three types of 3D data in a single figure
    
    Parameters:
    - scan_file: Path to point cloud scan data (.npy)
    - voxel_file: Path to voxel data (.ply)
    - feature_file: Path to feature data (.npz)
    
    Returns:
    - Matplotlib figure object
    """
    
    # Create figure for 3D data
    fig_3d = plt.figure(figsize=(24, 8))
    fig_3d.suptitle('3D Data Visualization Tool', fontsize=20, y=0.95)
    
    # ================== First Subplot: Point Cloud Scan Data ==================
    def visualize_scan(ax):
        # Load scan data
        if not os.path.exists(scan_file):
            print(f"Scan data file not found: {scan_file}")
            return
        
        scan_data = np.load(scan_file)
        print(f"\n[Point Cloud Data] Dimensions: {scan_data.ndim}D, Shape: {scan_data.shape}")
        
        # Process N×4 format data
        if scan_data.ndim == 2 and scan_data.shape[1] == 4:
            # First three columns are XYZ coordinates
            x = scan_data[:, 0]
            y = scan_data[:, 1]
            z = scan_data[:, 2]
            
            # Fourth column as color value
            colors = scan_data[:, 3]
            
            # Create scatter plot
            scatter = ax.scatter(x, y, z, 
                                 c=colors, 
                                 cmap='viridis',
                                 s=2,          # point size
                                 alpha=0.7)    # transparency
            
            ax.set_title('Point Cloud Scan Data (N×4 Format)', fontsize=14)

            # Set axis labels
            ax.set_xlabel('X Axis', fontsize=10, labelpad=5)
            ax.set_ylabel('Y Axis', fontsize=10, labelpad=5)
            ax.set_zlabel('Z Axis', fontsize=10, labelpad=5)

            # Set axis ticks
            ax.tick_params(axis='both', which='major', labelsize=8)

            # Add grid
            ax.grid(True, linestyle='--', alpha=0.3)

            # Set view angle
            ax.view_init(elev=25, azim=45)  # 25° elevation, 45° azimuth

            # Add legend
            ax.legend(['Points'], loc='upper right', fontsize=8)

            # Stretch Z-axis display ratio
            x_range = np.ptp(x)  # ptp = peak to peak = max - min
            y_range = np.ptp(y)
            z_range = np.ptp(z)
            scale_ratio = x_range / z_range if z_range > 0 else 1

            # Set same scale range
            max_range = max(x_range, y_range, z_range * scale_ratio)
            mid_x = (np.max(x) + np.min(x)) * 0.5
            mid_y = (np.max(y) + np.min(y)) * 0.5
            mid_z = (np.max(z) + np.min(z)) * 0.5

            ax.set_xlim(mid_x - max_range/2, mid_x + max_range/2)
            ax.set_ylim(mid_y - max_range/2, mid_y + max_range/2)
            ax.set_zlim(mid_z - max_range/2, mid_z + max_range/2)
            
            # Add data statistics
            stats = f"Point Count: {len(x)}\nX Range: [{x.min():.2f}, {x.max():.2f}]\nY Range: [{y.min():.2f}, {y.max():.2f}]\nZ Range: [{z.min():.2f}, {z.max():.2f}]"
            ax.text2D(0.05, 0.95, stats, transform=ax.transAxes, fontsize=8, 
                     verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
        else:
            ax.text(0.5, 0.5, f"Unsupported data format: {scan_data.shape}", 
                   horizontalalignment='center', verticalalignment='center',
                   transform=ax.transAxes, fontsize=12)
            ax.set_title('Point Cloud Data - Format Error', fontsize=14)

    # ================== Second Subplot: Voxel Data ==================
    def visualize_voxel(ax):
        # Load voxel data
        if not os.path.exists(voxel_file):
            print(f"Voxel data file not found: {voxel_file}")
            return
        
        try:
            # Read PLY file
            ply_data = plyfile.PlyData.read(voxel_file)
            vertices = ply_data['vertex']

            # Extract XYZ coordinates
            x = vertices['x']
            y = vertices['y']
            z = vertices['z']

            print(f"\n[Voxel Data] Vertex Count: {len(vertices)}")

            # Check for color information
            has_color = 'red' in vertices and 'green' in vertices and 'blue' in vertices
            has_intensity = 'intensity' in vertices or 'scalar_Intensity' in vertices

            # Prepare color data
            if has_color:
                # Extract RGB colors and normalize
                r = vertices['red'] / 255.0
                g = vertices['green'] / 255.0
                b = vertices['blue'] / 255.0
                colors = np.column_stack([r, g, b])
                color_mode = "RGB"
            elif has_intensity:
                # Extract intensity value
                intensity_key = 'intensity' if 'intensity' in vertices else 'scalar_Intensity'
                colors = vertices[intensity_key]
                color_mode = "Intensity"
            else:
                # Default to using Z coordinate as color
                colors = z
                color_mode = "Height"

            print(f"Color Mode: {color_mode}")

            # Create scatter plot
            if color_mode == "RGB":
                scatter = ax.scatter(x, y, z, 
                                     c=colors, 
                                     s=2,          # point size
                                     alpha=0.7)    # transparency
            else:
                scatter = ax.scatter(x, y, z, 
                                     c=colors, 
                                     cmap='viridis',
                                     s=2,          # point size
                                     alpha=0.7)    # transparency

            # Set title
            ax.set_title(f'Voxel Data ({color_mode})', fontsize=14)

            # Set axis labels
            ax.set_xlabel('X Axis', fontsize=10, labelpad=5)
            ax.set_ylabel('Y Axis', fontsize=10, labelpad=5)
            ax.set_zlabel('Z Axis', fontsize=10, labelpad=5)

            # Set axis ticks
            ax.tick_params(axis='both', which='major', labelsize=8)

            # Add grid
            ax.grid(True, linestyle='--', alpha=0.3)

            # Set view angle
            ax.view_init(elev=25, azim=45)  # 25° elevation, 45° azimuth

            # Add legend
            ax.legend(['Voxels'], loc='upper right', fontsize=8)

            # Stretch Z-axis display ratio
            x_range = np.ptp(x)  # ptp = peak to peak = max - min
            y_range = np.ptp(y)
            z_range = np.ptp(z)
            scale_ratio = x_range / z_range if z_range > 0 else 1

            # Set same scale range
            max_range = max(x_range, y_range, z_range * scale_ratio)
            mid_x = (np.max(x) + np.min(x)) * 0.5
            mid_y = (np.max(y) + np.min(y)) * 0.5
            mid_z = (np.max(z) + np.min(z)) * 0.5

            ax.set_xlim(mid_x - max_range/2, mid_x + max_range/2)
            ax.set_ylim(mid_y - max_range/2, mid_y + max_range/2)
            ax.set_zlim(mid_z - max_range/2, mid_z + max_range/2)
            
            # Add data statistics
            stats = f"Voxel Count: {len(x)}\nX Range: [{x.min():.2f}, {x.max():.2f}]\nY Range: [{y.min():.2f}, {y.max():.2f}]\nZ Range: [{z.min():.2f}, {z.max():.2f}]"
            if color_mode != "RGB":
                stats += f"\n{color_mode} Range: [{colors.min():.2f}, {colors.max():.2f}]"
            ax.text2D(0.05, 0.95, stats, transform=ax.transAxes, fontsize=8, 
                     verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
            
        except Exception as e:
            ax.text(0.5, 0.5, f"Error loading voxel data: {str(e)}", 
                   horizontalalignment='center', verticalalignment='center',
                   transform=ax.transAxes, fontsize=10)
            ax.set_title('Voxel Data - Loading Error', fontsize=14)

    # ================== Third Subplot: Feature Data ==================
    def visualize_features(ax):
        # Load feature data
        if not os.path.exists(feature_file):
            print(f"Feature data file not found: {feature_file}")
            return
        
        try:
            # Load NPZ file
            with np.load(feature_file) as data:
                print(f"\n[Feature Data] Arrays in file: {list(data.keys())}")
                
                # Get actual data
                indices = data['indices']        # voxel indices/coordinates
                patchtokens = data['patchtokens']  # feature vectors

            print(f"Indices Shape: {indices.shape}")
            print(f"Features Shape: {patchtokens.shape}")

            # Use indices as 3D coordinates
            x = indices[:, 0]  # X coordinate
            y = indices[:, 1]  # Y coordinate
            z = indices[:, 2]  # Z coordinate

            # Perform PCA dimensionality reduction from 128D to 3D
            print("Performing PCA dimensionality reduction...")
            pca = PCA(n_components=3)
            pca_result = pca.fit_transform(patchtokens)

            print(f"PCA Result Shape: {pca_result.shape}")
            print(f"Explained Variance Ratios: {pca.explained_variance_ratio_}")
            print(f"Cumulative Explained Variance: {np.sum(pca.explained_variance_ratio_):.3f}")

            # Normalize PCA results to [0,1] range for RGB values
            scaler = MinMaxScaler()
            rgb_values = scaler.fit_transform(pca_result)

            print(f"RGB Value Range: [{rgb_values.min():.3f}, {rgb_values.max():.3f}]")

            # Create scatter plot
            scatter = ax.scatter(x, y, z,
                                c=rgb_values,  # use normalized PCA results as RGB colors
                                s=10,          # point size
                                alpha=0.8,     # transparency
                                edgecolors='black',  # point edge color
                                linewidth=0.3) # edge line width

            # Set title
            ax.set_title('Feature Data (PCA Dimensionality Reduction)', fontsize=14)

            # Set axis labels
            ax.set_xlabel('X Coordinate', fontsize=10, labelpad=5)
            ax.set_ylabel('Y Coordinate', fontsize=10, labelpad=5)
            ax.set_zlabel('Z Coordinate', fontsize=10, labelpad=5)

            # Set equal aspect ratio
            x_range = np.ptp(x)
            y_range = np.ptp(y)
            z_range = np.ptp(z)
            max_range = max(x_range, y_range, z_range)

            # Calculate midpoints for each axis
            mid_x = (np.max(x) + np.min(x)) * 0.5
            mid_y = (np.max(y) + np.min(y)) * 0.5
            mid_z = (np.max(z) + np.min(z)) * 0.5

            # Set axis ranges while keeping center point
            ax.set_xlim(mid_x - max_range/2, mid_x + max_range/2)
            ax.set_ylim(mid_y - max_range/2, mid_y + max_range/2)
            ax.set_zlim(mid_z - max_range/2, mid_z + max_range/2)

            # Add grid
            ax.grid(True, linestyle='--', alpha=0.3)

            # Set view angle
            ax.view_init(elev=25, azim=45)

            # Add legend
            ax.legend(['Feature Points'], loc='upper right', fontsize=8)
            
            # Add data statistics
            stats = f"Feature Point Count: {len(x)}\nOriginal Feature Dimension: {patchtokens.shape[1]}\nCumulative Explained Variance: {np.sum(pca.explained_variance_ratio_):.3f}"
            ax.text2D(0.05, 0.95, stats, transform=ax.transAxes, fontsize=8, 
                     verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
            
            # Add color encoding explanation
            color_info = "Color Encoding:\n• Red (R): PCA Component 1\n• Green (G): PCA Component 2\n• Blue (B): PCA Component 3"
            ax.text2D(0.05, 0.75, color_info, transform=ax.transAxes, fontsize=8, 
                     verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
            
        except Exception as e:
            ax.text(0.5, 0.5, f"Error loading feature data: {str(e)}", 
                   horizontalalignment='center', verticalalignment='center',
                   transform=ax.transAxes, fontsize=10)
            ax.set_title('Feature Data - Loading Error', fontsize=14)

    # Create three subplots
    ax1 = fig_3d.add_subplot(131, projection='3d')
    visualize_scan(ax1)

    ax2 = fig_3d.add_subplot(132, projection='3d')
    visualize_voxel(ax2)

    ax3 = fig_3d.add_subplot(133, projection='3d')
    visualize_features(ax3)

    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # leave space for title
    plt.subplots_adjust(wspace=0.2)  # adjust subplot spacing
    
    return fig_3d

def display_map_info(row):
    """
    Display information for a map
    
    Parameters:
    - row: DataFrame row containing map information
    """
    print("=" * 80)
    print("Current Map Information:")
    print("=" * 80)
    
    print(f"Terrain Source (terrain_src): {row['terrain_src']}")
    print(f"Terrain Tag (terrain_tag): {row['terrain_tag']}")
    print(f"Patch File (patch_file): {row['patch_file']}")
    print(f"Patch ID (patch_id): {row['patch_id']}")
    print(f"Scan ID (scan_id): {row['scan_id']}")
    print(f"Voxel ID (voxel_id): {row['voxel_id']}")
    print(f"Feature ID (feat_id): {row['feat_id']}")
    
    print("\n" + "-" * 80)
    print("Converted Full File Paths:")
    print("-" * 80)
    
    print(f"EXR Source Folder (exr_source_folder):")
    print(f"  {row['exr_source_folder']}")
    
    print(f"\nScan File (scan_file):")
    print(f"  {row['scan_file']}")
    
    print(f"\nVoxel File (voxel_file):")
    print(f"  {row['voxel_file']}")
    
    print(f"\nFeature File (feature_file):")
    print(f"  {row['feature_file']}")
    
    print("\n" + "-" * 80)
    print("Relative Paths from Original CSV:")
    print("-" * 80)
    
    print(f"Scan Path (scan_path): {row['scan_path']}")
    print(f"Voxel Path (voxel_path): {row['voxel_path']}")
    print(f"Feature Path (full_feat_path): {row['full_feat_path']}")
    print(f"Render Directory (render_dir): {row['render_dir']}")
    
    # Check file existence
    print("\n" + "-" * 80)
    print("File Existence Check:")
    print("-" * 80)
    print(row['exr_source_folder'])
    files_to_check = [
        ('EXR Source Folder', row['exr_source_folder'], True),
        ('Scan File', row['scan_file'], False),
        ('Voxel File', row['voxel_file'], False),
        ('Feature File', row['feature_file'], False)
    ]
    
    for file_type, file_path, is_directory in files_to_check:
        if is_directory:
            exists = os.path.exists(file_path) and os.path.isdir(file_path)
        else:
            exists = os.path.exists(file_path)
        
        status = "Exists" if exists else "Does not exist"
        print(f"{file_type}: {status}")

def main():
    """Main function for terrain reconstruction visualization tool"""
    # CSV file path
    csv_path = "/media/wyc/Newsmy/sunxihe00/3_voxel/metadata.csv"
    
    # Check if file exists
    if not os.path.exists(csv_path):
        print(f"CSV file does not exist: {csv_path}")
        return
    
    try:
        # Read and convert CSV data
        print("Reading CSV file and converting path formats...")
        df = read_and_convert_csv(csv_path)
        
        if df is None or len(df) == 0:
            print("No valid data read")
            return
        
        print(f"Successfully converted {len(df)} rows of data")
        
        # Visualization parameters
        exr_exposure = 1.5
        exr_gamma = 2.2
        exr_normalize = True
        
        # Initialize figures
        fig_exr = None
        fig_3d = None
        
        # Main visualization loop
        while True:
            # Randomly select one row
            random_row = df.sample(n=1).iloc[0]
            
            # Display map information
            display_map_info(random_row)
            
            # Get paths for visualization
            exr_source_folder = random_row['exr_source_folder']
            scan_file = random_row['scan_file']
            voxel_file = random_row['voxel_file']
            feature_file = random_row['feature_file']
            
            # Close previous figures if they exist
            if fig_exr:
                plt.close(fig_exr)
            if fig_3d:
                plt.close(fig_3d)
            
            # Create both figures
            print("\nGenerating EXR visualization...")
            fig_exr = convert_and_display_random_12(
                exr_source_folder,
                exposure=exr_exposure,
                gamma=exr_gamma,
                normalize=exr_normalize
            )
            
            print("\nGenerating 3D data visualization...")
            fig_3d = visualize_3d_data(scan_file, voxel_file, feature_file)
            
            # Show both figures
            if fig_exr:
                plt.figure(fig_exr.number)
                plt.show(block=False)
            
            if fig_3d:
                plt.figure(fig_3d.number)
                plt.show(block=False)
            
            # Wait for user input
            print("\n" + "=" * 80)
            user_input = input("Press Enter to show another random map, or enter 'q' to quit: ").strip().lower()
            
            if user_input == 'q':
                print("Program exiting")
                break
            else:
                print("\n" + "=" * 80)
                print("Loading next map...")
                print("=" * 80)
                
    except Exception as e:
        print(f"Error processing CSV file: {e}")
    finally:
        # Close all figures on exit
        plt.close('all')

if __name__ == "__main__":
    main()