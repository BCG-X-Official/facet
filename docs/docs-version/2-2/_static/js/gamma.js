$(document).ready(function() {
    $('a.reference.external').attr('target', '_blank');
    DOCUMENTATION_OPTIONS.VERSION = DOCS_VERSIONS.current;
    buildVersionSelector();
});


const buildVersionSelector = function() {

    const versionDropdown = $('<div class="navbar-end-item"><select/></div>');
    const versionSelector = versionDropdown.children()

    versionSelector.change(function() {
        navigateToDocsVersion($(this).val());
    });

    const activeDocsVersion = getActiveDocsVersion()

    DOCS_VERSIONS.all.forEach(function(docsVersion) {
        versionSelector
            .append($('<option/>')
                .html("Version: " + docsVersion)
                .attr("value", docsVersion)
                .attr("selected", docsVersion.startsWith(activeDocsVersion))
            );
    });

    $("#navbar-end").append(versionDropdown);
}

const getActiveDocsVersion = function() {
    const currentLocation = window.location + "";
    if (currentLocation.indexOf("docs-version") === -1) {
        return DOCS_VERSIONS.current;
    } else {
        const rExp = /.*docs-version\/(\d-\d.*)\/.*/g;
        const matches = rExp.exec(currentLocation);
        if (matches && matches.length > 1) {
            // convert back from URL to real version string
            return matches[1].split("-").join(".");
        } else{
            return DOCS_VERSIONS.current;
        }
    }
}

const navigateToDocsVersion = function(targetVersion) {
    const currentLocation = window.location + "";
    const subUrl = "docs-version/" + targetVersion.split(".").slice(0, 2).join("-") + "/index.html";

    if (currentLocation.indexOf("docs-version/") > 0) {
        const startIndex = currentLocation.indexOf("docs-version/")
        window.location = currentLocation.substring(0, startIndex) + subUrl
    } else {
        window.location = subUrl
    }
}
