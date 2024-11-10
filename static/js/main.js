document.addEventListener('DOMContentLoaded', function() {
    const contentInput = document.getElementById('content-input');
    const styleInput = document.getElementById('style-input');
    const transferBtn = document.getElementById('transfer-btn');
    const loading = document.getElementById('loading');

    const MAX_FILE_SIZE = 16 * 1024 * 1024; // 16MB

    function validateFile(file) {
        if (file.size > MAX_FILE_SIZE) {
            alert('File size must be less than 16MB');
            return false;
        }
        return true;
    }

    function previewImage(input, previewId) {
        const preview = document.getElementById(previewId);
        const file = input.files[0];
        
        if (file) {
            const reader = new FileReader();
            reader.onload = function(e) {
                const img = `<img src="${e.target.result}" alt="Preview">`;
                preview.innerHTML = img;
                
                // Add click event to the new image
                preview.querySelector('img').onclick = function() {
                    showModal(this.src);
                };
            }
            reader.readAsDataURL(file);
        }
    }

    contentInput.addEventListener('change', function() {
        previewImage(this, 'content-preview');
    });

    styleInput.addEventListener('change', function() {
        previewImage(this, 'style-preview');
    });

    transferBtn.addEventListener('click', async function() {
        if (!contentInput.files[0] || !styleInput.files[0]) {
            alert('Please select both content and style images');
            return;
        }

        if (!validateFile(contentInput.files[0]) || !validateFile(styleInput.files[0])) {
            return;
        }

        const formData = new FormData();
        formData.append('content', contentInput.files[0]);
        formData.append('style', styleInput.files[0]);

        loading.style.display = 'block';
        transferBtn.disabled = true;

        try {
            const response = await fetch('/transfer', {
                method: 'POST',
                body: formData
            });

            const data = await response.json();

            if (data.error) {
                alert(data.error);
            } else {
                const resultPreview = document.getElementById('result-preview');
                resultPreview.innerHTML = `<img src="${data.result}" alt="Result">`;
                // Add click handler to the result image
                resultPreview.querySelector('img').onclick = function() {
                    showModal(this.src);
                };
            }
        } catch (error) {
            alert('An error occurred during style transfer');
            console.error(error);
        } finally {
            loading.style.display = 'none';
            transferBtn.disabled = false;
        }
    });

    // Add modal HTML to body
    const modalHTML = `
        <div id="imageModal" class="modal">
            <div class="modal-content">
                <span class="close-modal">&times;</span>
                <img class="modal-image" id="modalImage" src="" alt="Preview">
            </div>
        </div>
    `;
    document.body.insertAdjacentHTML('beforeend', modalHTML);

    // Get modal elements
    const modal = document.getElementById('imageModal');
    const modalImage = document.getElementById('modalImage');
    const closeModal = document.querySelector('.close-modal');

    function showModal(imageSrc) {
        modalImage.src = imageSrc;
        modal.style.display = 'flex';
    }

    // Close modal when clicking close button or outside
    closeModal.onclick = () => modal.style.display = 'none';
    modal.onclick = (e) => {
        if (e.target === modal) modal.style.display = 'none';
    };

    // Add click handler for result preview
    document.getElementById('result-preview').addEventListener('click', function(e) {
        if (e.target.tagName === 'IMG') {
            showModal(e.target.src);
        }
    });
}); 