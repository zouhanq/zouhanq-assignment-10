document.addEventListener("DOMContentLoaded", () => {
    const searchForm = document.getElementById("searchForm");
    const resultImages = document.getElementById("resultImages");

    // Handle form submission
    searchForm.addEventListener("submit", async (event) => {
        event.preventDefault(); // Prevent default form submission behavior

        // Get form data
        const formData = new FormData(searchForm);

        // Clear previous results
        resultImages.innerHTML = "<p>Loading results...</p>";

        try {
            // Send the POST request to the backend
            const response = await fetch("/search", {
                method: "POST",
                body: formData,
            });

            // Handle errors
            if (!response.ok) {
                throw new Error("Failed to fetch results. Please try again.");
            }

            // Parse the JSON response
            const results = await response.json();

            // Display the results
            displayResults(results);
        } catch (error) {
            console.error("Error:", error);
            resultImages.innerHTML = `<p class="error">${error.message}</p>`;
        }
    });

    // Function to display results
    function displayResults(results) {
        // Clear the results container
        resultImages.innerHTML = "";

        if (results.length === 0) {
            resultImages.innerHTML = "<p>No results found.</p>";
            return;
        }

        // Create and append image elements for each result
        results.forEach((result) => {
            const imgWrapper = document.createElement("div");
            imgWrapper.classList.add("result-item");

            const img = document.createElement("img");
            img.src = `/results/${result.file_name}`;
            img.alt = `Similarity score: ${result.score.toFixed(2)}`;
            img.title = `Similarity score: ${result.score.toFixed(2)}`;
            imgWrapper.appendChild(img);

            const score = document.createElement("p");
            score.textContent = `Score: ${result.score.toFixed(2)}`;
            imgWrapper.appendChild(score);

            resultImages.appendChild(imgWrapper);
        });
    }
});
