# Paid parking reservations

The driver page now shows each configured car (`C01`–`C10`) and motorcycle
(`M01`–`M20`) slot. Green means the camera reports vacant and no checkout or
reservation claims it. Yellow means a checkout temporarily holds the slot or a
paid reservation is active. Red means the camera detects a vehicle. Camera
offline/stale is shown as Unknown and cannot be reserved.

The reservation fee is the **full prepaid parking rate** for the selected day,
week, or month, calculated by the backend's existing pricing rules. It is not
an additional fee. GCash/PayMongo must be configured for online reservation.

1. The driver selects a green slot and duration, then starts a GCash checkout.
2. Postgres creates a 15-minute `pending_payment` hold. A partial unique index
   prevents another checkout or reservation from claiming that slot.
3. Only after PayMongo confirms payment does one transaction record the
   parking payment and activate the reservation. The driver's account/email,
   slot, amount, duration, start, and expiry are saved.
4. A detected vehicle turns the slot red and changes the reservation to
   `occupied`. The camera **cannot identify whose vehicle it is**; an
   unauthorized vehicle in a reserved slot requires staff intervention.
5. Once the slot is observed empty for 10 seconds after occupancy, the session
   completes and the reservation releases. Unused reservations expire at the
   end of their paid duration. Physical occupancy remains red even after
   reservation expiry, never falsely green.

If a PayMongo confirmation arrives **after the 15-minute hold expires**, or
reservation activation otherwise fails, the checkout is marked as an issue at
`/api/gcash/issues` for administrator refund/reassignment. It is never silently
shown as an active reservation. Keep the PayMongo webhook enabled so a paid
checkout is confirmed even when the driver closes the success page.

Deploy the updated backend to create the reservation table and checkout
column. Both camera workers must keep sending fresh per-slot zone updates.
The feature does not open slots for booking while camera status is unknown.
