// Excel export links: download in the background so the chat stays on screen, and show progress or the
// server's (safe, short) error message next to the link. Without JavaScript the link still works as a normal
// download. Messages are set with textContent only.
(() => {
  function filenameFrom(header) {
    const match = /filename="([^"]+)"/.exec(header || "");
    return match ? match[1] : null;
  }

  async function download(link) {
    const status = link.parentElement ? link.parentElement.querySelector(".export-status") : null;
    const say = (text, isError) => {
      if (!status) return;
      status.textContent = text;
      status.classList.toggle("error", Boolean(isError));
    };
    link.setAttribute("aria-busy", "true");
    say("Preparing the workbook…", false);
    try {
      const response = await fetch(link.href, { credentials: "same-origin" });
      if (!response.ok) {
        let detail = "The export failed. The result is still shown above.";
        try {
          const body = await response.json();
          if (body && typeof body.detail === "string") detail = body.detail;
        } catch (ignored) {
          // not JSON: keep the generic message
        }
        say(detail, true);
        return;
      }
      const blob = await response.blob();
      const url = URL.createObjectURL(blob);
      const anchor = document.createElement("a");
      anchor.href = url;
      anchor.download = filenameFrom(response.headers.get("Content-Disposition")) || "export.xlsx";
      document.body.appendChild(anchor);
      anchor.click();
      anchor.remove();
      setTimeout(() => URL.revokeObjectURL(url), 10000);
      say("Downloaded.", false);
    } catch (error) {
      say("The export failed (network problem). Try again.", true);
    } finally {
      link.removeAttribute("aria-busy");
    }
  }

  document.addEventListener("click", (event) => {
    const target = event.target;
    const link = target && typeof target.closest === "function" ? target.closest("a.export-link") : null;
    if (!link || event.defaultPrevented || event.button !== 0 || event.ctrlKey || event.metaKey || event.shiftKey) return;
    event.preventDefault();
    if (link.getAttribute("aria-busy") !== "true") download(link);
  });
})();
