const form = document.getElementById("generate-form");

if (form) {
    const generateBtn = document.getElementById("generate-btn");
    const resultCard = document.getElementById("result-card");
    const resultContent = document.getElementById("result-content");
    const errorCard = document.getElementById("error-card");
    const copyBtn = document.getElementById("copy-btn");

    form.addEventListener("submit", async (e) => {
        e.preventDefault();

        const topic = document.getElementById("topic").value.trim();
        const platform = document.getElementById("platform").value;
        const tone = document.getElementById("tone").value;

        errorCard.classList.add("hidden");
        resultCard.classList.add("hidden");

        generateBtn.disabled = true;
        generateBtn.textContent = "Generating...";

        try {
            const response = await fetch("/generate", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ topic, platform, tone }),
            });

            const data = await response.json();

            if (!response.ok) {
                throw new Error(data.error || "Something went wrong.");
            }

            resultContent.textContent = data.content;
            resultCard.classList.remove("hidden");
        } catch (err) {
            errorCard.textContent = err.message;
            errorCard.classList.remove("hidden");
        } finally {
            generateBtn.disabled = false;
            generateBtn.textContent = "Generate";
        }
    });

    if (copyBtn) {
        copyBtn.addEventListener("click", () => {
            navigator.clipboard.writeText(resultContent.textContent);
            copyBtn.textContent = "Copied!";
            setTimeout(() => (copyBtn.textContent = "Copy"), 1500);
        });
    }
}