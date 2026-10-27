(howto-recover-alignment)=
# How to save work in progress and recover it

This guide shows you how to save an alignment that is not ready to upload, so it is not lost if the
GUI is closed or crashes.

## Steps

1. Click **Save Progress**, or press {kbd}`Shift+S`. With more than one shank, choose which shanks
   to save.

   The alignment is written to `alignment_progress.json` in the output folder
   (`alignment_progress_shank<N>.json` for a multi-shank probe).

2. Next time the session is opened, the saved alignment is loaded automatically. It is listed in
   the alignment dropdown as `recovered (<time saved>)`, so it cannot be mistaken for an uploaded
   alignment.

   ```{image} ../../_static/how_to/align/recovered.png
   :alt: The alignment dropdown open, with the recovered alignment and the time it was saved at the top of the list
   :align: center
   ```

3. Carry on aligning, and upload when ready. Uploading deletes the progress file.

## Related

- {ref}`howto-resume-alignment`
- {ref}`ref-output-files`
