from flask import Flask, render_template, request, send_from_directory
import os
from search import ImageSearcher

app = Flask(__name__)

# Initialize searcher
searcher = ImageSearcher()

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        query_type = request.form.get('query_type', 'text')  # text, image, hybrid
        text_query = request.form.get('text_query', '').strip()
        lam_str = request.form.get('lambda', '0.5')
        lam = float(lam_str) if lam_str else 0.5
        use_pca = request.form.get('use_pca') == 'on'

        # Handle image query if provided
        image_file = request.files.get('image_query')
        
        query_emb = None
        
        if query_type == 'text':
            # Text-only query
            if not text_query:
                return render_template('index.html', error="Please provide a text query.")
            query_emb = searcher.encode_text(text_query)
            
        elif query_type == 'image':
            # Image-only query
            if not image_file or image_file.filename == '':
                return render_template('index.html', error="Please provide an image query.")
            query_emb = searcher.encode_image(image_file)
            
        elif query_type == 'hybrid':
            # Hybrid query requires both text and image
            if not text_query:
                return render_template('index.html', error="Please provide a text query for hybrid search.")
            if not image_file or image_file.filename == '':
                return render_template('index.html', error="Please provide an image for hybrid search.")
            text_emb = searcher.encode_text(text_query)
            img_emb = searcher.encode_image(image_file)
            query_emb = searcher.combine_embeddings(text_emb, img_emb, lam)
        else:
            return render_template('index.html', error="Invalid query type.")
        
        # Now we have query_emb, run the search
        results = searcher.search_similar(query_emb, top_k=5, use_pca=use_pca)

        # results is a list of (filename, score)
        # Prepend the image folder to display images
        displayed_results = []
        for fname, score in results:
            # The images are located in coco_images_resized
            image_path = os.path.join('coco_images_resized', fname)
            displayed_results.append((fname, score, image_path))

        return render_template('index.html', 
                               results=displayed_results,
                               query_type=query_type,
                               text_query=text_query,
                               lambda_val=lam,
                               use_pca_checked='checked' if use_pca else '',
                               message="Search complete")

    return render_template('index.html')

# If you need to serve images from the folder
@app.route('/coco_images_resized/<path:filename>')
def serve_image(filename):
    return send_from_directory('coco_images_resized', filename)


if __name__ == '__main__':
    app.run(debug=True)
