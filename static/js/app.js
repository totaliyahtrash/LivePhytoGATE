/**
 * PhytoGATE Diagnostic Core — Frontend Application Controller
 * 
 * Strict architectural rules enforced:
 * 1. Exactly ONE POST /api/diagnose per user action.
 * 2. Complete previous state clearing before launching a diagnosis.
 * 3. Never display fake percentages or stale metrics on WITHHELD results.
 * 4. Distinct presentation for Cache Hits vs Live Inferences.
 * 5. Vendor-neutral product presentation (PhytoGATE Vision Core).
 */

document.addEventListener("DOMContentLoaded", () => {
  // DOM Elements
  const dropzone = document.getElementById("dropzone");
  const fileInput = document.getElementById("file-input");
  const previewContainer = document.getElementById("preview-container");
  const previewImg = document.getElementById("preview-img");
  const previewFilename = document.getElementById("preview-filename");
  const previewFilesize = document.getElementById("preview-filesize");
  const btnDiagnose = document.getElementById("btn-diagnose");
  const btnReset = document.getElementById("btn-reset");

  // Output containers
  const standbyState = document.getElementById("standby-state");
  const loadingState = document.getElementById("loading-state");
  const loadingText = document.getElementById("loading-text");
  const verdictContainer = document.getElementById("verdict-container");
  
  // Primary Verdict
  const primaryVerdictCard = document.getElementById("primary-verdict-card");
  const verdictBadge = document.getElementById("verdict-badge");
  const badgeCached = document.getElementById("badge-cached");
  const diagnosisHeaderLabel = document.getElementById("diagnosis-header-label");
  const verdictDiagnosisTitle = document.getElementById("verdict-diagnosis-title");
  const metricHost = document.getElementById("metric-host");
  const metricAssessment = document.getElementById("metric-assessment");
  const metricConfidenceItem = document.getElementById("metric-confidence-item");
  const metricConfidenceDesc = document.getElementById("metric-confidence-desc");
  const metricOrigin = document.getElementById("metric-origin");
  
  // Evidence & Differential
  const sectionEvidence = document.getElementById("section-evidence");
  const evidenceList = document.getElementById("evidence-list");
  const sectionDifferential = document.getElementById("section-differential");
  const alternativeDiagnosisText = document.getElementById("alternative-diagnosis-text");
  const sectionLimitations = document.getElementById("section-limitations");
  const limitationsList = document.getElementById("limitations-list");
  const sectionWithheldReason = document.getElementById("section-withheld-reason");
  const withheldReasonText = document.getElementById("withheld-reason-text");

  // Telemetry items
  const telStatus = document.getElementById("tel-status");
  const telModel = document.getElementById("tel-model");
  const telCache = document.getElementById("tel-cache");

  // Active state
  let currentFile = null;
  let currentPreviewUrl = null;
  let activeVisualizations = null;

  // Initialize
  fetchTelemetry();

  // -------------------------------------------------------------
  // Drag & Drop / File Selection
  // -------------------------------------------------------------
  dropzone.addEventListener("click", () => fileInput.click());

  dropzone.addEventListener("dragover", (e) => {
    e.preventDefault();
    dropzone.classList.add("dragover");
  });

  dropzone.addEventListener("dragleave", () => {
    dropzone.classList.remove("dragover");
  });

  dropzone.addEventListener("drop", (e) => {
    e.preventDefault();
    dropzone.classList.remove("dragover");
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      handleSelectedFile(e.dataTransfer.files[0]);
    }
  });

  fileInput.addEventListener("change", (e) => {
    if (e.target.files && e.target.files.length > 0) {
      handleSelectedFile(e.target.files[0]);
    }
  });

  btnReset.addEventListener("click", () => {
    resetAllState();
  });

  function handleSelectedFile(file) {
    if (!file.type.startsWith("image/") && !file.name.match(/\.(jpg|jpeg|png|webp|bmp|tif|tiff)$/i)) {
      alert("Please upload a standard image file (JPG, JPEG, PNG, WEBP).");
      return;
    }

    currentFile = file;
    if (currentPreviewUrl) {
      URL.revokeObjectURL(currentPreviewUrl);
    }
    currentPreviewUrl = URL.createObjectURL(file);

    // Update preview
    previewImg.src = currentPreviewUrl;
    previewFilename.textContent = file.name;
    previewFilesize.textContent = `${(file.size / 1024).toFixed(1)} KB`;
    previewContainer.style.display = "flex";
    btnDiagnose.disabled = false;

    // Reset previous diagnostic results while preserving preview
    clearDiagnosticOutput();
  }

  // -------------------------------------------------------------
  // Safe Telemetry
  // -------------------------------------------------------------
  async function fetchTelemetry() {
    try {
      const res = await fetch("/api/telemetry");
      if (!res.ok) return;
      const data = await res.json();
      
      telStatus.textContent = data.status || "ONLINE";
      telStatus.className = "telemetry-value telemetry-status-ok";
      
      // Sanitized user-facing product name
      telModel.textContent = "PhytoGATE Vision Core";

      if (data.cache) {
        telCache.textContent = `${data.cache.cached_entries_count} entries (${data.cache.total_cache_hits} hits)`;
      }
    } catch (err) {
      telStatus.textContent = "OFFLINE";
      telStatus.className = "telemetry-value";
      telStatus.style.color = "var(--status-crimson)";
    }
  }

  // -------------------------------------------------------------
  // State Clearing (Zero stale values)
  // -------------------------------------------------------------
  function clearDiagnosticOutput() {
    standbyState.style.display = "flex";
    loadingState.style.display = "none";
    verdictContainer.style.display = "none";

    // Clear text values
    if (diagnosisHeaderLabel) diagnosisHeaderLabel.textContent = "DETECTED DISEASE";
    verdictDiagnosisTitle.textContent = "—";
    verdictDiagnosisTitle.className = "diagnosis-name";
    if (metricHost) metricHost.textContent = "—";
    metricAssessment.textContent = "—";
    metricAssessment.className = "metric-val";
    if (metricConfidenceItem) metricConfidenceItem.style.display = "none";
    if (metricConfidenceDesc) metricConfidenceDesc.textContent = "—";
    metricOrigin.textContent = "PhytoGATE Vision Core";

    evidenceList.innerHTML = "";
    sectionEvidence.style.display = "none";

    alternativeDiagnosisText.textContent = "";
    sectionDifferential.style.display = "none";

    limitationsList.innerHTML = "";
    sectionLimitations.style.display = "none";

    sectionWithheldReason.style.display = "none";
    withheldReasonText.textContent = "";
  }

  function resetAllState() {
    clearDiagnosticOutput();
    currentFile = null;
    if (currentPreviewUrl) {
      URL.revokeObjectURL(currentPreviewUrl);
      currentPreviewUrl = null;
    }
    previewImg.src = "";
    previewContainer.style.display = "none";
    btnDiagnose.disabled = true;
    fileInput.value = "";
  }

  // -------------------------------------------------------------
  // Execute Diagnostic Flow
  // -------------------------------------------------------------
  btnDiagnose.addEventListener("click", async () => {
    if (!currentFile) return;

    // 1. Clear ALL previous state before launching request
    clearDiagnosticOutput();
    standbyState.style.display = "none";
    loadingState.style.display = "flex";
    btnDiagnose.disabled = true;
    loadingText.textContent = "PHYTOGATE DIAGNOSTIC CORE • ANALYZING SPECIMEN...";

    const formData = new FormData();
    formData.append("file", currentFile);

    try {
      // Exactly ONE POST /api/diagnose
      const response = await fetch("/api/diagnose", {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        throw new Error(`Server returned status HTTP ${response.status}`);
      }

      const result = await response.json();
      renderVerdict(result);
    } catch (err) {
      console.error("Diagnosis request error:", err);
      renderNetworkError(err.message);
    } finally {
      loadingState.style.display = "none";
      btnDiagnose.disabled = false;
      fetchTelemetry(); // Update cache count
    }
  });

  // -------------------------------------------------------------
  // Render Structured Diagnostic Result
  // -------------------------------------------------------------
  function renderVerdict(result) {
    // CRITICAL: Complete wipe of any prior UI state before populating new response
    clearDiagnosticOutput();
    standbyState.style.display = "none";
    loadingState.style.display = "none";
    verdictContainer.style.display = "flex";

    activeVisualizations = result.visualizations || {};

    if (result.status === "SUCCESS") {
      // ----------------- SUCCESS STATE -----------------
      primaryVerdictCard.className = "card verdict-card";
      verdictBadge.textContent = "CONFIRMED";
      verdictBadge.className = "status-badge badge-success";

      if (result.is_cached) {
        badgeCached.style.display = "inline-flex";
        badgeCached.className = "status-badge badge-cached";
        badgeCached.textContent = "DETERMINISTIC CACHE HIT";
        metricOrigin.textContent = "Deterministic Cache";
      } else {
        badgeCached.style.display = "none";
        metricOrigin.textContent = "PhytoGATE Vision Core";
      }

      // Check if genuinely classified as healthy
      const isHealthy = (result.disease && result.disease.toLowerCase() === "healthy") || 
                        (result.diagnosis && result.diagnosis.toLowerCase().includes("healthy"));

      if (isHealthy) {
        if (diagnosisHeaderLabel) diagnosisHeaderLabel.textContent = "PLANT STATUS";
        verdictDiagnosisTitle.textContent = "HEALTHY";
        verdictDiagnosisTitle.className = "diagnosis-name status-healthy";
      } else {
        if (diagnosisHeaderLabel) diagnosisHeaderLabel.textContent = "DETECTED DISEASE";
        // Extract pure disease name - NOT host prefix
        let diseaseName = result.disease;
        if (!diseaseName && result.diagnosis) {
          diseaseName = result.diagnosis.includes("—") ? result.diagnosis.split("—")[1].trim() : result.diagnosis;
        }
        verdictDiagnosisTitle.textContent = diseaseName || "Unknown Condition";
        verdictDiagnosisTitle.className = "diagnosis-name";
      }

      if (metricHost) {
        metricHost.textContent = result.host || "Unknown";
      }
      
      const assess = (result.assessment || "unknown").toUpperCase();
      metricAssessment.textContent = assess;
      metricAssessment.className = `metric-val val-${result.assessment || 'unknown'}`;

      if (result.confidence_score !== null && result.confidence_score !== undefined) {
        if (metricConfidenceItem) metricConfidenceItem.style.display = "flex";
        if (metricConfidenceDesc) metricConfidenceDesc.textContent = `${(result.confidence_score * 100).toFixed(1)}%`;
      } else {
        if (metricConfidenceItem) metricConfidenceItem.style.display = "none";
        if (metricConfidenceDesc) metricConfidenceDesc.textContent = "—";
      }

      // Evidence
      if (result.visual_evidence && result.visual_evidence.length > 0) {
        sectionEvidence.style.display = "block";
        evidenceList.innerHTML = "";
        result.visual_evidence.forEach(item => {
          const li = document.createElement("li");
          li.textContent = item;
          evidenceList.appendChild(li);
        });
      } else {
        sectionEvidence.style.display = "none";
        evidenceList.innerHTML = "";
      }

      // Alternative diagnosis
      if (result.alternative_diagnosis) {
        sectionDifferential.style.display = "block";
        alternativeDiagnosisText.textContent = result.alternative_diagnosis;
      } else {
        sectionDifferential.style.display = "none";
        alternativeDiagnosisText.textContent = "";
      }

      // Limitations
      if (result.limitations && result.limitations.length > 0) {
        sectionLimitations.style.display = "block";
        limitationsList.innerHTML = "";
        result.limitations.forEach(lim => {
          const li = document.createElement("li");
          li.textContent = lim;
          limitationsList.appendChild(li);
        });
      } else {
        sectionLimitations.style.display = "none";
        limitationsList.innerHTML = "";
      }

    } else {
      // ----------------- WITHHELD STATE -----------------
      primaryVerdictCard.className = "card verdict-card";
      verdictBadge.textContent = "DIAGNOSIS WITHHELD";
      verdictBadge.className = "status-badge badge-withheld";
      badgeCached.style.display = "none";

      if (diagnosisHeaderLabel) diagnosisHeaderLabel.textContent = "STATUS";
      verdictDiagnosisTitle.textContent = "DIAGNOSIS WITHHELD";
      verdictDiagnosisTitle.className = "diagnosis-name status-withheld";

      if (metricHost) {
        metricHost.textContent = result.host || "Unconfirmed";
      }
      metricAssessment.textContent = "WITHHELD";
      metricAssessment.className = "metric-val val-unknown";
      if (metricConfidenceItem) metricConfidenceItem.style.display = "none";
      if (metricConfidenceDesc) metricConfidenceDesc.textContent = "None";
      metricOrigin.textContent = result.error_category || "INSPECTION_WITHHELD";

      // Withheld rationale
      sectionWithheldReason.style.display = "block";
      withheldReasonText.textContent = result.error_message || "The vision service could not complete the analysis. No diagnosis was generated. Please try again later.";

      // Evidence or limitations if provided by model
      if (result.visual_evidence && result.visual_evidence.length > 0) {
        sectionEvidence.style.display = "block";
        evidenceList.innerHTML = "";
        result.visual_evidence.forEach(item => {
          const li = document.createElement("li");
          li.textContent = item;
          evidenceList.appendChild(li);
        });
      } else {
        sectionEvidence.style.display = "none";
        evidenceList.innerHTML = "";
      }

      if (result.limitations && result.limitations.length > 0) {
        sectionLimitations.style.display = "block";
        limitationsList.innerHTML = "";
        result.limitations.forEach(lim => {
          const li = document.createElement("li");
          li.textContent = lim;
          limitationsList.appendChild(li);
        });
      } else {
        sectionLimitations.style.display = "none";
        limitationsList.innerHTML = "";
      }

      sectionDifferential.style.display = "none";
      alternativeDiagnosisText.textContent = "";
    }
  }

  function renderNetworkError(errorMessage) {
    clearDiagnosticOutput();
    standbyState.style.display = "none";
    loadingState.style.display = "none";
    verdictContainer.style.display = "flex";

    primaryVerdictCard.className = "card verdict-card";
    verdictBadge.textContent = "DIAGNOSIS WITHHELD";
    verdictBadge.className = "status-badge badge-withheld";
    badgeCached.style.display = "none";

    if (diagnosisHeaderLabel) diagnosisHeaderLabel.textContent = "STATUS";
    verdictDiagnosisTitle.textContent = "DIAGNOSIS WITHHELD";
    verdictDiagnosisTitle.className = "diagnosis-name status-withheld";

    if (metricHost) {
      metricHost.textContent = "Unconfirmed";
    }
    metricAssessment.textContent = "WITHHELD";
    metricAssessment.className = "metric-val val-unknown";
    if (metricConfidenceItem) metricConfidenceItem.style.display = "none";
    if (metricConfidenceDesc) metricConfidenceDesc.textContent = "None";
    metricOrigin.textContent = "SERVICE_UNAVAILABLE";

    sectionWithheldReason.style.display = "block";
    withheldReasonText.textContent = `Communication error: ${errorMessage}. The system refuses to fabricate a fallback diagnosis.`;
  }

});
