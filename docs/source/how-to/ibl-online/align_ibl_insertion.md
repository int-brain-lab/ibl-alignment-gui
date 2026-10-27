(howto-align-ibl-insertion)=
# How to align an IBL insertion from Alyx

This guide shows you how to load a probe insertion from the IBL Alyx database, align it, and upload
the alignment and its quality assessment back to Alyx.

## Prerequisites

- The `ibl` extra installed (see {ref}`howto-install-extras`).
- An Alyx account, with ONE set up for it. The first time ONE is used it asks for the database
  URL and your credentials.

## Steps

1. Start the GUI in IBL mode. To go straight to one insertion, pass its probe insertion id:

   ```console
   alignment-gui-ibl -p <pid>
   ```

   Without `-p`, choose the subject, the session and probe, and the shank from the dropdowns, and
   click **Load**. Only insertions with spike sorting are listed.

2. Align as usual. The data is downloaded through ONE and cached locally, so later loads are
   faster.

3. Upload ({kbd}`Shift+U`). For each shank, a QC dialog asks for:

   - the confidence of the alignment: High, Medium or Low
   - the QC of the recording: Pass, Warning or Critical, with at least one problem ticked for
     Warning or Critical
   - whether to resolve the insertion's alignment with this one

   With more than one shank, tick **Apply to all shanks** to give the remaining shanks the same
   assessment. Cancelling the dialog cancels the upload.

4. The channel locations, the alignment and the assessment are saved to Alyx, under your Alyx user
   name.

Previous alignments of the insertion, by any user, are in the alignment dropdown (see
{ref}`howto-resume-alignment`).

## Related

- {ref}`ref-command-line`: `alignment-gui-ibl` options
- {ref}`ref-plugins`: the QC dialog
