import streamlit.components.v1 as components

def razorpay_button():
    components.html(
        """
        <form>
            <script src="https://checkout.razorpay.com/v1/payment-button.js" data-payment_button_id="pl_LeZt2rbF4z2CqX" async></script>
        </form>
        """,
        height=500,
    )
