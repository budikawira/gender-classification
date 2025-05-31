<h1>Gender Classification</h1>
<p>
  This project demonstrates the training and evaluation of multiple AI models using various Convolutional Neural Network (CNN) architectures to classify gender from image data.
</p>

<h2>📂 Project Files</h2>
<ul>
  <li><strong>gender_classification.ipynb</strong><br>
      Jupyter Notebook used for training the AI models using different CNN architectures.</li>
  <li><strong>gender_classification_evaluation.ipynb</strong><br>
      Jupyter Notebook containing evaluation reports, visualizations, and comparison metrics between trained models.</li>
</ul>

<h2>🛠 Features</h2>
<ul>
  <li>Multiple CNN architectures implemented</li>
  <li>Model performance comparison (accuracy, loss, etc.)</li>
  <li>Evaluation metrics including confusion matrix, precision, recall, F1-score</li>
  <li>Visualization of training history</li>
  <li>Easy-to-read evaluation summary</li>
</ul>

<h2>🚀 Getting Started</h2>

<h3>Prerequisites</h3>
<p>Make sure the following packages are installed:</p>

<pre><code>pandas
scikit-learn
torch
torchvision
Pillow
matplotlib
imagehash
</code></pre>

<p>You can install them using:</p>

<pre><code>pip install pandas
pip install scikit-learn
pip install torch
pip install torchvision
pip install Pillow
pip install matplotlib
pip install imagehash
</code></pre>

<h3>How to Run</h3>
<ol>
  <li>Clone the repository:
    <pre><code>git clone https://github.com/yourusername/gender-classification.git
cd gender-classification</code></pre>
  </li>
  <li>Open the training notebook:
    <pre><code>jupyter notebook gender_classification.ipynb</code></pre>
  </li>
  <li>The <code>gender_classification.ipynb</code> notebook is prepared to automatically download the dataset. However, this step can be skipped if you prefer to manually place the dataset in the <strong>Dataset</strong> folder.</li>
  <li>This experiment uses a subset of the CelebA dataset, consisting of approximately 5,000 images. The dataset may require exploration and cleanup before training to ensure high-quality results.</li>
  <li>After training, open the evaluation notebook:
    <pre><code>jupyter notebook gender_classification_evaluation.ipynb</code></pre>
  </li>
</ol>

<h2>📈 Results</h2>
<p>The evaluation notebook provides detailed results and comparisons between all trained models using standard classification metrics.</p>
<p>After running <code>gender_classification.ipynb</code>, the following folders will be generated:</p>
<ul>
  <li><strong>Dataset</strong> – Contains the processed data and images used for training and testing.</li>
  <li><strong>Evaluation</strong> – Stores training and evaluation logs, including metrics and performance summaries.</li>
  <li><strong>Models</strong> – Saves the best model states (weights) for each CNN architecture trained.</li>
</ul>

<h2>📝 License</h2>
<p>This project is for educational and research purposes only.</p>
