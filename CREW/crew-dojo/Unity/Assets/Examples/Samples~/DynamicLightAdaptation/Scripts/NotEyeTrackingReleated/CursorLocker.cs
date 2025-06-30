//-----------------------------------------------------------------------
// Copyright 2016 Tobii AB (publ). All rights reserved.
//-----------------------------------------------------------------------

using UnityEngine;

namespace Tobii.Gaming.Examples.DynamicLightAdaptation
{
	public class CursorLocker : MonoBehaviour
	{
		public bool LockCursor = true;

		void Update()
		{
			if (Input.GetKeyUp(KeyCode.Escape))
			{
				LockCursor = !LockCursor;
			}

			UpdateCursor();
		}

		private void UpdateCursor()
		{
			if (LockCursor)
			{
				Cursor.lockState = CursorLockMode.Locked;
				Cursor.visible = false;
			}
			else if (!LockCursor)
			{
				Cursor.lockState = CursorLockMode.None;
				Cursor.visible = true;
			}
		}

		void OnDestroy()
		{
			Cursor.lockState = CursorLockMode.None;
			Cursor.visible = true;
		}
	}
}
