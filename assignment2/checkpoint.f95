module debug

implicit none;

! =================================================================
!
! This module implements a checkpoint function with optional 
!  arguments to allow writing to file
! 
! -----------------------------------------------------------------
!
! SUBROUTINES:
!
! checkpoint(text, debug, filepointer, filename)
!
!
! Inputs
! | text (character) The message to be printed
!
! | debug (logical) If false, printing is disabled
!
! | filepointer (integer) (optional) If present, the message is 
! written into the file this points to 
!
! | filename (character) (optional) If present and filepointer not
! present, a file is opened at this path; if the file exists 
! already, the text is appended at the end
 

contains

subroutine checkpoint (text, debug, filepointer, filename)
    character (len=*), intent(in) :: text
    logical, intent(in) :: debug
    character (len=*), intent(in), optional :: filename
    integer, intent(in), optional :: filepointer
    logical :: file_exists
    
    if (.not. debug) then
        return ! if debug is false, do nothing
    end if
    
    if (present(filepointer)) then
        write (filepointer, *) text
        ! write to the filepointer
    else if (present(filename)) then
        inquire(file=filename, exist=file_exists)
        if (file_exists) then
            open(99, file=filename, status="old", access="append")
            ! open the already existing file in append mode
        else
            open(99, file=filename, status="new")
            ! create and open a new file
        end if
    
        write (99, *) text
        ! write to the file
        
        close(99)
        ! close the file
    else 
        write(0,*) text
        ! if no file is given, write to stderr
    end if

end subroutine

end module
