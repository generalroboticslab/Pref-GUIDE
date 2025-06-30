//-----------------------------------------------------------------------
// Copyright 2016 Tobii AB (publ). All rights reserved.
//-----------------------------------------------------------------------

using UnityEngine;
using UnityEngine.UI;

namespace Tobii.Gaming.Examples.UserPresenceAndHeadPose
{
	public class UserPresenceStatus : MonoBehaviour
	{
		public Text TextViewUserPresenceStatus;
		public Text TextViewIsUserPresent;

		void Update()
		{
			UpdateUserPresenceView();
		}

		/// <summary>
		/// Print the User Presence status
		/// </summary>
		private void UpdateUserPresenceView()
		{
			var userPresence = TobiiAPI.GetUserPresence();
			TextViewUserPresenceStatus.text = userPresence.ToString();

			if (TobiiAPI.GetUserPresence().IsUserPresent())
			{
				TextViewIsUserPresent.text = "Yes";
			}
			else
			{
				TextViewIsUserPresent.text = "No";
			}
		}
	}
}
