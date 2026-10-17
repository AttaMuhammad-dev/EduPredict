document.addEventListener(
    "DOMContentLoaded",
    function () {


        // =========================
        // PREDICTION
        // =========================

        const predictionForm =
            document.getElementById(
                "predictForm"
            );


        if (predictionForm) {

            predictionForm.addEventListener(
                "submit",
                async function (event) {

                    event.preventDefault();


                    const result =
                        document.getElementById(
                            "predictionResult"
                        );


                    result.innerHTML = `
                        <div class="loading-result">
                            <div class="loader"></div>
                            <h3>Analyzing...</h3>
                            <p>
                                Machine learning model
                                is processing the data.
                            </p>
                        </div>
                    `;


                    const formData =
                        new FormData(
                            predictionForm
                        );


                    const payload = {

                        attendance:
                            Number(
                                formData.get(
                                    "attendance"
                                )
                            ),

                        marks:
                            Number(
                                formData.get(
                                    "marks"
                                )
                            ),

                        lms_activity:
                            Number(
                                formData.get(
                                    "lms_activity"
                                )
                            ),

                        previous_performance:
                            Number(
                                formData.get(
                                    "previous_performance"
                                )
                            )

                    };


                    try {

                        const response =
                            await fetch(
                                "/api/predict",
                                {
                                    method: "POST",

                                    headers: {
                                        "Content-Type":
                                            "application/json"
                                    },

                                    body:
                                        JSON.stringify(
                                            payload
                                        )
                                }
                            );


                        const data =
                            await response.json();


                        if (!data.success) {

                            throw new Error(
                                data.error
                            );

                        }


                        let riskClass =
                            "low";


                        if (
                            data.risk_level
                                === "Medium Risk"
                        ) {

                            riskClass =
                                "medium";

                        }


                        if (
                            data.risk_level
                                === "High Risk"
                        ) {

                            riskClass =
                                "high";

                        }


                        result.innerHTML = `

                            <div class="prediction-success">

                                <div class="
                                    prediction-badge
                                    ${riskClass}
                                ">

                                    ${data.risk_level}

                                </div>


                                <div class="
                                    confidence-number
                                ">

                                    ${data.confidence}%

                                </div>


                                <p>
                                    Model confidence
                                </p>


                                <div class="recommendation">

                                    <strong>
                                        Recommendation
                                    </strong>

                                    <p>
                                        ${data.recommendation}
                                    </p>

                                </div>

                            </div>

                        `;

                    }


                    catch (error) {

                        result.innerHTML = `

                            <div class="error-result">

                                ${error.message}

                            </div>

                        `;

                    }

                }
            );

        }


        // =========================
        // CSV UPLOAD
        // =========================

        const uploadForm =
            document.getElementById(
                "uploadForm"
            );


        if (uploadForm) {

            uploadForm.addEventListener(
                "submit",
                async function (event) {

                    event.preventDefault();


                    const result =
                        document.getElementById(
                            "uploadResult"
                        );


                    const formData =
                        new FormData(
                            uploadForm
                        );


                    result.innerHTML =
                        "Uploading dataset...";


                    try {

                        const response =
                            await fetch(
                                "/api/upload",
                                {
                                    method: "POST",
                                    body: formData
                                }
                            );


                        const data =
                            await response.json();


                        if (!data.success) {

                            throw new Error(
                                data.error
                            );

                        }


                        result.innerHTML = `

                            <div class="
                                success-result
                            ">

                                ✓
                                ${data.message}

                            </div>

                        `;

                    }

                    catch (error) {

                        result.innerHTML = `

                            <div class="
                                error-result
                            ">

                                ${error.message}

                            </div>

                        `;

                    }

                }
            );

        }


        // =========================
        // STUDENT SEARCH
        // =========================

        const search =
            document.getElementById(
                "studentSearch"
            );


        if (search) {

            search.addEventListener(
                "input",
                function () {

                    const query =
                        search.value
                            .toLowerCase();


                    const rows =
                        document.querySelectorAll(
                            "#studentTable tbody tr"
                        );


                    rows.forEach(
                        function (row) {

                            const text =
                                row.textContent
                                    .toLowerCase();


                            row.style.display =
                                text.includes(
                                    query
                                )
                                    ? ""
                                    : "none";

                        }
                    );

                }
            );

        }

    }
);


// =========================
// CONTACT FORM
// =========================

function submitContact(event) {

    event.preventDefault();


    alert(
        "Your support request has been submitted successfully."
    );


    event.target.reset();

}